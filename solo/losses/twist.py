# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch


def twist_loss_func(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Computes TWIST's loss given batch of class logits p1 from view 1 and
    class logits p2 from view 2.

    Args:
        p1 (torch.Tensor): NxC Tensor containing class logits from view 1.
        p2 (torch.Tensor): NxC Tensor containing class logits from view 2.

    Returns:
        torch.Tensor: TWIST's loss.
    """
    eps = 1e-6

    p1_softmax = p1.softmax(dim=1)
    p1_logsoftmax = p1.log_softmax(dim=1)
    p2_softmax = p2.softmax(dim=1)
    p2_logsoftmax = p2.log_softmax(dim=1)

    # first term
    kl_div = (
        (p2_softmax * p2_logsoftmax).sum(dim=1) - (p2_softmax * p1_logsoftmax).sum(dim=1)
    ).mean()

    # second term
    mean_entropy = -(p1_softmax * (p1_logsoftmax + eps)).sum(dim=1).mean()

    # third term
    mean_prob = p1_softmax.mean(dim=0)
    entropy_mean = -(mean_prob * (mean_prob.log() + eps)).sum()

    return kl_div + mean_entropy - entropy_mean
