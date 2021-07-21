import torch

from solo.losses import direct_pred_loss_func


def test_direct_pred_loss():
    b, f = 32, 128
    p = torch.randn(b, f).requires_grad_()
    z = torch.randn(b, f)

    loss = direct_pred_loss_func(p, z)
    initial_loss = loss.item()
    assert loss != 0

    for i in range(20):
        loss = direct_pred_loss_func(p, z)
        loss.backward()
        p.data.add_(-0.5 * p.grad)

        p.grad = None

    assert loss < initial_loss
