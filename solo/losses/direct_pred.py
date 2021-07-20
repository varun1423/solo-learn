import torch
import torch.nn.functional as F


def direct_pred_loss_func(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Computes DirectPred's loss given batch of predicted features p and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2

    Returns:
        torch.Tensor: DirectPred's loss.
    """

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -2 * (p * z.detach()).sum(dim=-1).mean()
