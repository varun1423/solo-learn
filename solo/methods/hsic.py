import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
from solo.losses.hsic import hsic_loss_func
from solo.methods.base import BaseMomentumModel
from solo.utils.momentum import initialize_momentum_params


class HSIC(BaseMomentumModel):
    def __init__(
        self, output_dim, proj_hidden_dim, pred_hidden_dim, **kwargs,
    ):
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
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

        self.bn_helper = BatchNorm1d(output_dim, affine=False, track_running_stats=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = super(HSIC, HSIC).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")

        # projector
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        return parent_parser

    @property
    def learnable_params(self):
        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self):
        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X, *args, **kwargs):
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        feats1_momentum, feats2_momentum = out["feats_momentum"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # forward momentum encoder
        with torch.no_grad():
            z1_momentum = self.momentum_projector(feats1_momentum)
            z2_momentum = self.momentum_projector(feats2_momentum)

        indexes = batch[0]
        labels = F.one_hot(torch.arange(p1.size(0), device=self.device)).to(torch.float32)
        kernel_params = torch.linalg.norm(labels.unsqueeze(1) - labels, dim=-1)

        p1 = F.normalize(self.bn_helper(p1), dim=-1)
        p2 = F.normalize(self.bn_helper(p2), dim=-1)
        z1_momentum = F.normalize(self.bn_helper(z1_momentum), dim=-1)
        z2_momentum = F.normalize(self.bn_helper(z2_momentum), dim=-1)

        p = torch.cat((p1, p2))
        z_momentum = torch.cat((z1_momentum, z2_momentum))

        # define kernel matrices
        hiddens = [p1, z2_momentum]
        # K = torch.mm(p, z_momentum.T)
        # L = torch.eye(p1.size(0) * 2, device=self.device, dtype=torch.float32)
        # L[:, p1.size(0) :].fill_diagonal_(1.0)
        # L[p1.size(0) :, :].fill_diagonal_(1.0)
        # ------- contrastive loss -------
        hsic_loss = hsic_loss_func(hiddens, kernel_params)

        # calculate std of features
        z_std = (F.normalize(z1).std(dim=0).mean() + F.normalize(z2).std(dim=0).mean()) / 2

        metrics = {
            "train_hsic_loss": hsic_loss,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return hsic_loss + class_loss
