import argparse
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.methods.base import BaseMethod


class Interpolate(nn.Module):
    def __init__(self, upscale: str = "scale", size: Optional[int] = None):
        super().__init__()
        self.upscale = upscale
        self.size = size

        if self.upscale == "size":
            assert self.size is not None

    def forward(self, x):
        if self.upscale == "scale":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        elif self.upscale == "size":
            return F.interpolate(x, size=(self.size, self.size), mode="nearest")


def conv3x3(in_planes: int, out_planes: int, groups: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, groups=groups, bias=True)


def conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        upscale: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.upsample = upsample
        self.upscale = upscale

    def forward(self, x) -> torch.Tensor:
        identity = x

        # if upscale is not None it will also be added to upsample
        out = x
        if self.upscale is not None:
            out = self.upscale(out)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        upscale: Optional[nn.Module] = None,
    ) -> None:
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, groups)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.upscale = upscale

    def forward(self, x) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        # if upscale is not None it will also be added to upsample
        if self.upscale is not None:
            out = self.upscale(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        block: Union[BasicBlock, Bottleneck],
        layers: List[int],
        input_height: int = 32,
        latent_dim: int = 128,
        h_dim: int = 2048,
        groups: int = 1,
        widen: int = 1,
        width_per_group: int = 512,
        cifar: bool = False,
        remove_first_maxpool: bool = False,
    ) -> None:

        super(Decoder, self).__init__()

        self.cifar = cifar
        self.remove_first_maxpool = remove_first_maxpool
        self.upscale_factor = 8

        if not cifar:
            self.upscale_factor *= 2

        if not remove_first_maxpool:
            self.upscale_factor *= 2

        self.input_height = input_height
        self.h_dim = h_dim
        self.groups = groups
        self.inplanes = h_dim
        self.base_width = 64
        num_out_filters = width_per_group * widen

        self.linear_projection1 = nn.Linear(latent_dim, h_dim, bias=True)
        self.linear_projection2 = nn.Linear(h_dim, h_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(self.h_dim // 16, self.h_dim)

        num_out_filters /= 2
        self.layer1 = self._make_layer(
            block,
            int(num_out_filters),
            layers[0],
            Interpolate(upscale="size", size=self.input_height // self.upscale_factor),
        )
        num_out_filters /= 2
        self.layer2 = self._make_layer(block, int(num_out_filters), layers[1], Interpolate())
        num_out_filters /= 2
        self.layer3 = self._make_layer(block, int(num_out_filters), layers[2], Interpolate())
        num_out_filters /= 2
        self.layer4 = self._make_layer(block, int(num_out_filters), layers[3], Interpolate())

        self.conv2 = conv3x3(int(num_out_filters) * block.expansion, self.base_width)
        self.final_conv = conv3x3(self.base_width, 3)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        upscale: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        upsample = None

        if self.inplanes != planes * block.expansion or upscale is not None:
            # this is passed into residual block for skip connection
            upsample = []
            if upscale is not None:
                upsample.append(upscale)
            upsample.append(conv1x1(self.inplanes, planes * block.expansion))
            upsample = nn.Sequential(*upsample)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                upsample,
                self.groups,
                self.base_width,
                upscale,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    upscale=None,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.linear_projection1(x))
        x = self.relu(self.linear_projection2(x))

        x = x.view(x.size(0), self.h_dim // 16, 4, 4)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.remove_first_maxpool:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = self.conv2(x)
        x = self.relu(x)

        if not self.cifar:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = self.final_conv(x)

        return x


def decoder18(**kwargs):
    # layers list is opposite the encoder (in this case [2, 2, 2, 2])
    return Decoder(BasicBlock, [2, 2, 2, 2], **kwargs)


def decoder34(**kwargs):
    # layers list is opposite the encoder (in this case [3, 6, 4, 3])
    return Decoder(BasicBlock, [3, 6, 4, 3], **kwargs)


def decoder50(**kwargs):
    # layers list is opposite the encoder
    return Decoder(Bottleneck, [3, 6, 4, 3], **kwargs)


def decoder50w2(**kwargs):
    # layers list is opposite the encoder
    return Decoder(Bottleneck, [3, 6, 4, 3], widen=2, **kwargs)


def decoder50w4(**kwargs):
    # layers list is opposite the encoder
    return Decoder(Bottleneck, [3, 6, 4, 3], widen=4, **kwargs)


class ProjectorVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, output_dim, bias=False)
        self.logvar = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return self.mu(x), self.logvar(x)


class AAVAE(BaseMethod):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        decoder_hidden_dim: int,
        kl_coeff: float,
        log_scale: float,
        **kwargs
    ):
        """Implements AAVAE (https://arxiv.org/abs/2107.12329).

        Args:
            output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            # TODO
        """

        super().__init__(**kwargs)

        self.kl_coeff = kl_coeff
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]))

        # projector
        self.projector = ProjectorVAE(self.features_dim, proj_hidden_dim, output_dim)

        # decoder
        dataset = self.extra_args["dataset"]
        if dataset in ["cifar10", "cifar100"]:
            input_size = 32
        elif dataset == "stl10":
            input_size = 96
        elif dataset in ["imagenet100", "imagenet"]:
            input_size = 224

        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        decoder_model = {"resnet18": decoder18, "resnet50": decoder50}[self.encoder_name]
        self.decoder = decoder_model(
            input_height=input_size,
            latent_dim=output_dim,
            h_dim=decoder_hidden_dim,
            cifar=cifar,
            remove_first_maxpool=cifar,
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(AAVAE, AAVAE).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("aavae")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # decoder
        parser.add_argument("--decoder_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--log_scale", type=float, default=0.0)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.decoder.parameters()},
        ]
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
        mu, log_var = self.projector(out["feats"])
        return {**out, "mu": mu, "log_var": log_var}

    def sample(self, z_mu, z_var, eps=1e-6):
        # add eps to prevent 0 variance
        std = torch.exp(z_var / 2.0) + eps

        p = torch.distributions.Normal(torch.zeros_like(z_mu), torch.ones_like(std))
        q = torch.distributions.Normal(z_mu, std)
        z = q.rsample()

        return p, q, z

    @staticmethod
    def kl_divergence_analytic(p, q, z):
        kl = torch.distributions.kl.kl_divergence(q, p).sum(dim=-1)
        log_pz = p.log_prob(z).sum(dim=-1)
        log_qz = q.log_prob(z).sum(dim=-1)

        return kl, log_pz, log_qz

    @staticmethod
    def gaussian_likelihood(mean, logscale, sample, eps=1e-6):
        scale = torch.exp(logscale / 2.0) + eps
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)

        # sum over dimensions
        return log_pxz.sum(dim=(1, 2, 3))

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR and supervised SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.n_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        original_imgs = batch[1][0]
        _, c, h, w = original_imgs.size()
        pixels = c * h * w

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        feats_orig, feats_aug = out["feats"]

        with torch.no_grad():
            feats_orig = feats_orig.detach()
            mu_orig, log_var_orig = self.projector(feats_orig)

        mu, log_var = self.projector(feats_aug)

        p, q, z = self.sample(mu, log_var)
        kl, log_pz, log_qz = self.kl_divergence_analytic(p, q, z)

        log_pzs = []
        log_qzs = []
        log_pxzs = []

        kls = []
        elbos = []
        losses = []
        cos_sims = []
        kl_augmentations = []

        for _ in range(1):
            p, q, z = self.sample(mu, log_var)
            kl, log_pz, log_qz = self.kl_divergence_analytic(p, q, z)

            with torch.no_grad():
                _, q_orig, z_orig = self.sample(mu_orig, log_var_orig)

            # kl between original image and augmented image
            kl_aug = torch.distributions.kl.kl_divergence(q, q_orig).sum(dim=-1)
            kl_augmentations.append(kl_aug)

            cos_sims.append(self.cosine_similarity(z_orig, z))

            x_hat = self.decoder(z)
            log_pxz = self.gaussian_likelihood(x_hat, self.log_scale, original_imgs)

            # plot reconstructions
            # img_grid = torchvision.utils.make_grid(x_hat)

            elbo = kl - log_pxz
            loss = self.kl_coeff * kl - log_pxz

            log_pzs.append(log_pz)
            log_qzs.append(log_qz)
            log_pxzs.append(log_pxz)

            kls.append(kl)
            elbos.append(elbo)
            losses.append(loss)

        # all of these will be of shape [batch, samples, ... ]
        log_pz = torch.stack(log_pzs, dim=1)
        log_qz = torch.stack(log_qzs, dim=1)
        log_pxz = torch.stack(log_pxzs, dim=1)

        kl = torch.stack(kls, dim=1)
        elbo = torch.stack(elbos, dim=1).mean()
        aavae_loss = torch.stack(losses, dim=1).mean()

        cos_sim = torch.stack(cos_sims, dim=1).mean()
        kl_augmentation = torch.stack(kl_augmentations, dim=1).mean()

        log_px = torch.logsumexp(log_pxz + log_pz - log_qz, dim=1).mean(dim=0)

        bpd = -log_px / (pixels * torch.log(torch.tensor(2)))  # need log_px in base 2

        metrics = {
            "train_kl": kl.mean(),
            "train_elbo": elbo,
            "train_aavae_loss": aavae_loss,
            "train_bpd": bpd,
            "train_cos_sim": cos_sim,
            "train_kl_augmentation": kl_augmentation,
            "train_log_pxz": log_pxz.mean(),
            "train_log_pz": log_pz.mean(),
            "train_log_px": log_px,
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return aavae_loss + class_loss
