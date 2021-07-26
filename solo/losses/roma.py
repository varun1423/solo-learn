import torch
import torch.nn.functional as F


def roma_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
    random_matrix: torch.Tensor,
    temperature: float = 0.5,
    gamma: float = 1,
    lamb: float = 8,
) -> torch.Tensor:
    """Computes ROMA's loss given batch of projected features z1 from view 1,
    projected features z2 from view 2 and negative projected features z3 from another image.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        z3 (torch.Tensor): NxD Tensor containing projected features from another image.
        random_matrix (torch.Tensor): DxD' Tensor to project the features into a random space.
        temperature (float): temperature factor for the loss. Defaults to 0.5.
        gamma (float): margin of the triplet loss. Defaults to 1.0.
        lamb (float): weights the importance of both losses. Defaults to 8.0.

    Returns:
        torch.Tensor: ROMA loss.
    """

    z1 = F.normalize(z1 @ random_matrix, dim=-1)
    z2 = F.normalize(z2 @ random_matrix, dim=-1)
    z3 = F.normalize(z3 @ random_matrix, dim=-1)

    pos_sim = (z1 @ z2.T).sum(dim=1)  # positive pair similarity
    neg_sim = (z1 @ z3.T).sum(dim=1)  # negative pair similarity

    triplet_loss = torch.clamp(gamma + neg_sim - pos_sim, min=0.0).mean()

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1) / temperature
    targets = torch.zeros(z1.size(0), device=z1.device, dtype=torch.long)
    ce_loss = F.cross_entropy(logits, targets)

    return triplet_loss + lamb * ce_loss
