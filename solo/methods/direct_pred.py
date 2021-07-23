import argparse
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.direct_pred import direct_pred_loss_func as dp_loss_func
from solo.methods.base import BaseMomentumModel
from solo.utils.momentum import initialize_momentum_params


class Accumulator:
    def __init__(self, dp_rho=None):
        """
        Added typing to Accumulator from
        https://github.com/facebookresearch/luckmatters/blob/master/ssl/real-dataset/byol_trainer.py
        """

        self.dp_rho = dp_rho
        self.cumulated = None
        self.counter = 0

        self.reset()

    def reset(self):
        """Resets statistics from accumulator."""

        if self.dp_rho is None:
            # Averaging..
            self.cumulated = None
            self.counter = 0

    def add_list(self, d_list: list):
        """Adds a list of batches of features to the current accumulator.

        Args:
            d_list (list): list of batches of features.
        """

        assert isinstance(d_list, list)

        all_d = torch.cat(d_list, dim=0)
        if all_d.size(0) == 0:
            d = torch.zeros(*all_d.size()[1:]).to(device=all_d.get_device())
        else:
            d = all_d.mean(dim=0)

        self.add(d)

    def add(self, d: torch.Tensor):
        """Adds a batch of features to the current estimation.

        Args:
            d (torch.Tensor): batch of features.
        """

        if self.cumulated is None:
            self.cumulated = d
        else:
            if self.dp_rho is None:
                self.cumulated += d
            else:
                self.cumulated = self.dp_rho * self.cumulated + (1 - self.dp_rho) * d

        self.counter += 1

    def get(self) -> torch.Tensor:
        """Computes the mean of the accumulated features.

        Returns:
            torch.Tensor: mean of the accumulated features.
        """

        if self.dp_rho is None:
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
        dp_eps: float,
        dp_rho: float,
        **kwargs,
    ):
        """Implements DirecPred (https://arxiv.org/abs/2006.07733).

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            predictor_update_freq (int) number of steps between each update of the predictor.
            dp_eps: eps value for DirectPred in Eq. 18.
            dp_rho: rho value for DirectPred in Eq. 19.
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
        self.dp_eps = dp_eps
        self.dp_rho = dp_rho

        self.cum_corr = Accumulator(dp_rho=self.dp_rho)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(DirectPred, DirectPred).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("direct_pred")

        # projector
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        parser.add_argument("--predictor_update_freq", type=int, default=1)
        parser.add_argument("--dp_eps", type=float, default=0.01)
        parser.add_argument("--dp_rho", type=float, default=0.3)

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

    def compute_w_corr(self, M: torch.Tensor) -> torch.Tensor:
        """Computes W given correlation matrix.
        Adapted from:
            https://github.com/facebookresearch/luckmatters/blob/master/ssl/real-dataset/byol_trainer.py

        Args:
            M (torch.Tensor): covariance matrix.

        Returns:
            torch.Tensor: new weights for the linear predictor.
        """

        # we need to convert because torch.eig doesn't support half precision
        M = M.to(torch.float32)
        D, Q = torch.eig(M, eigenvectors=True)

        # if eigen_values >= 1, scale everything down.
        max_eig = D[:, 0].max()
        eigen_values = D[:, 0].clamp(0) / max_eig
        # Going through a concave function (sqrt function)
        # to boost small eigenvalues (while still keep very small one to be 0)
        # Note that here dp_eps is allowed to be negative.
        eigen_values = eigen_values.sqrt() + self.dp_eps
        eigen_values = eigen_values.clamp(1e-4)

        w = Q @ eigen_values.diag() @ Q.t()
        w = w.to(torch.half)
        return w

    @torch.no_grad()
    def update_predictor(self, batch_idx: int):
        """Updates the predictor via DirectPred by performing eigen-decomposition

        Args:
            batch_idx (int): current batch index to see if needs to do an update.
        """

        if batch_idx % self.predictor_update_freq == 0:
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
        corrs = []
        for z in (z1_detach, z2_detach):
            corr = torch.bmm(z.unsqueeze(2), z.unsqueeze(1))
            corrs.append(corr)

        self.cum_corr.add_list(corrs)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # forward momentum encoder
        with torch.no_grad():
            z1_momentum = self.momentum_projector(feats1_momentum)
            z2_momentum = self.momentum_projector(feats2_momentum)

        # ------- contrastive loss -------
        neg_cos_sim = dp_loss_func(p1, z2_momentum) + dp_loss_func(p2, z1_momentum)
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

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int, dataloader_idx: int
    ):
        # update predictor via DirectPred
        self.update_predictor(batch_idx)
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
