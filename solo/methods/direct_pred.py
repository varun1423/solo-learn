import argparse
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.direct_pred import direct_pred_loss_func as dp_loss_func
from solo.methods.base import BaseMomentumModel
from solo.utils.momentum import initialize_momentum_params


class Accumulator:
    def __init__(self, dyn_lambda=None):
        """
        Copied From
        https://github.com/facebookresearch/luckmatters/blob/master/ssl/real-dataset/byol_trainer.py
        """

        self.dyn_lambda = dyn_lambda
        self.cumulated = None
        self.counter = 0

        self.reset()

    def reset(self):
        if self.dyn_lambda is None:
            # Averaging..
            self.cumulated = None
            self.counter = 0

    def add_list(self, d_list):
        assert isinstance(d_list, list)

        all_d = torch.cat(d_list, dim=0)
        if all_d.size(0) == 0:
            d = torch.zeros(*all_d.size()[1:]).to(device=all_d.get_device())
        else:
            d = all_d.mean(dim=0)

        self.add(d)

    def add(self, d):
        if self.cumulated is None:
            self.cumulated = d
        else:
            if self.dyn_lambda is None:
                self.cumulated += d
            else:
                self.cumulated = self.dyn_lambda * self.cumulated + (1 - self.dyn_lambda) * d

        self.counter += 1

    def get(self):
        if self.dyn_lambda is None:
            assert self.counter > 0
            return self.cumulated / self.counter
        else:
            return self.cumulated.clone()


class DirectPred(BaseMomentumModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        predictor_update_freq: int,
        dyn_eps: float,
        dyn_lambda: float,
        dyn_convert: int,
        **kwargs,
    ):
        """Implements DirecPred BYOL (https://arxiv.org/abs/2006.07733).

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
        """

        super().__init__(**kwargs)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Linear(output_dim, output_dim, bias=False)
        for param in self.predictor.parameters():
            param.requires_grad = False

        self.predictor_update_freq = predictor_update_freq
        self.dyn_eps = dyn_eps
        self.dyn_lambda = dyn_lambda
        self.dyn_convert = dyn_convert

        self.cum_corr = Accumulator(dyn_lambda=self.dyn_lambda)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(DirectPred, DirectPred).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("direct_pred")

        # projector
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        parser.add_argument("--predictor_update_freq", type=int, default=1)
        parser.add_argument("--dyn_eps", type=float, default=0.01)
        parser.add_argument("--dyn_lambda", type=float, default=0.3)
        parser.add_argument("--dyn_convert", type=float, default=2)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online encoder (encoder, projector and predictor).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def compute_w_corr(self, M):
        """Compute W given correlation matrix.
        Code copied from:
            https://github.com/facebookresearch/luckmatters/blob/master/ssl/real-dataset/byol_trainer.py

        Args:
            M ([type]): [description]

        Raises:
            RuntimeError: [description]

        Returns:
            [type]: [description]
        """
        M = M.to(torch.float32)
        D, Q = torch.eig(M, eigenvectors=True)

        # if eigen_values >= 1, scale everything down.
        max_eig = D[:, 0].max()
        eigen_values = D[:, 0].clamp(0) / max_eig
        # Going through a concave function (dyn_convert > 1, e.g., 2 or sqrt function)
        # to boost small eigenvalues (while still keep very small one to be 0)
        # Note that here dyn_eps is allowed to be negative.
        eigen_values = eigen_values.pow(1 / self.dyn_convert) + self.dyn_eps
        eigen_values = eigen_values.clamp(1e-4)

        w = Q @ eigen_values.diag() @ Q.t()
        w = w.to(torch.half)
        return w

    def update_predictor(self, batch_idx):
        if batch_idx == 0 or batch_idx % self.predictor_update_freq == 0:
            M = self.cum_corr.get()
            if M is not None:
                w = self.compute_w_corr(M)
                self.predictor.weight.data.copy_(w)

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for BYOL reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        feats1_momentum, feats2_momentum = out["feats_momentum"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # compute stuff for predictor
        z1_detach = z1.detach()
        z2_detach = z2.detach()
        corrs, means = [], []
        for z in (z1_detach, z2_detach):
            corr = torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
            corrs.append(corr)
            means.append(z)
        self.cum_corr.add_list(corrs)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # forward momentum encoder
        with torch.no_grad():
            z1_momentum = self.momentum_projector(feats1_momentum)
            z2_momentum = self.momentum_projector(feats2_momentum)

        # ------- contrastive loss -------
        neg_cos_sim = dp_loss_func(p1, z2_momentum) + dp_loss_func(p2, z1_momentum)

        # update predictor via DirectPred
        self.update_predictor(batch_idx)

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
