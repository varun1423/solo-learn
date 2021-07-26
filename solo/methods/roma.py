import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.losses.roma import roma_loss_func
from solo.methods.base import BaseModel


class ROMA(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        gamma: float,
        lamb: float,
        random_projection_dim: int,
        **kwargs
    ):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the loss.
            gamma (float): margin of the triplet loss.
            lamb (float): weights the importance of both losses.
            random_projection_dim (int): number of dimensions of the random projection.
        """

        super().__init__(**kwargs)

        self.temperature = temperature
        self.gamma = gamma
        self.lamb = lamb

        self.random_projection_size = (output_dim, random_projection_dim)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(proj_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(ROMA, ROMA).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("roma")

        # projector
        parser.add_argument("--output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # random matrix
        parser.add_argument("--random_projection_dim", type=int, default=1024)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.5)
        parser.add_argument("--gamma", type=float, default=1.0)
        parser.add_argument("--lamb", type=float, default=8)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the encoder, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for ROMA reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes, *_, target = batch

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)
        # negative pair for triplet loss
        z3 = z1[range(-1, z1.size(0))]

        # prepare random matrix
        random_matrix = torch.randn(self.random_projection_size, device=self.device)

        # ------- self-supervised loss -------
        triplet_loss = roma_loss_func(
            z1,
            z2,
            z3,
            random_matrix,
            temperature=self.temperature,
            gamma=self.gamma,
            lamb=self.lamb,
        )

        metrics = {
            "train_triplet_loss": triplet_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return triplet_loss + class_loss
