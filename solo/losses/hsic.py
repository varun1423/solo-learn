import math

import mpmath
import numpy as np
import torch


def hsic_loss_func(hiddens, kernel_param, num_rff_features=512, gamma=3):
    hsic_yz = compute_hsic_yz(hiddens, kernel_param, num_rff_features)
    hsic_zz = compute_hsic_zz(hiddens, kernel_param, num_rff_features)
    return -hsic_yz + gamma * torch.sqrt(hsic_zz)


def compute_hsic_yz(hiddens, kernel_param, num_rff_features):
    device = hiddens[0].device
    b = hiddens[0].size(0)
    M = len(hiddens)

    rff_hiddens = torch.zeros((b, num_rff_features), device=device)
    mean = torch.zeros((1, num_rff_features))

    for hidden in hiddens:
        rff_features = imq_rff_features(hidden, num_rff_features, kernel_param)
        rff_hiddens += rff_features
        mean += rff_features.sum(0, keepdims=True)

    return (rff_hiddens ** 2).sum() / (b * M * (M - 1)) - (mean ** 2).sum() / (b * M) ** 2


def compute_hsic_zz(hiddens, kernel_param, num_rff_features):
    device = hiddens[0].device
    b = hiddens[0].size(0)
    M = len(hiddens)

    z1_rffs = []
    z2_rffs = []
    center_z1 = torch.zeros((1, num_rff_features), device=device)
    center_z2 = torch.zeros((1, num_rff_features), device=device)

    for hidden in hiddens:
        z1_rff = imq_rff_features(hidden, num_rff_features, kernel_param)
        z1_rffs.append(z1_rff)
        center_z1 += z1_rff.mean(0, keepdims=True)

        z2_rff = imq_rff_features(hidden, num_rff_features, kernel_param)
        z2_rffs.append(z2_rff)
        center_z2 += z2_rff.mean(0, keepdims=True)
    center_z1 /= M
    center_z2 /= M

    z = torch.zeros((num_rff_features, num_rff_features), device=device)
    for z1_rff, z2_rff in zip(z1_rffs, z2_rffs):
        z += torch.einsum("ni,nj->ij", z1_rff - center_z1, z2_rff - center_z2)

    return (z ** 2).sum() / (b * M - 1) ** 2


def imq_rff_features(hidden, num_rff_features, kernel_param):
    device = hidden.device
    d = hidden.size(-1)
    pi = torch.tensor(math.pi, device=device)

    amp, amp_probs = amplitude_frequency_and_probs(d, device)
    amplitudes = torch.from_numpy(
        np.random.choice(amp, size=[num_rff_features, 1], p=amp_probs)
    ).to(device)
    directions = torch.normal((num_rff_features, d)).to(device)
    b = torch.rand(size=(1, num_rff_features), device=device) * 2 * pi
    w = directions / torch.linalg.norm(directions, axis=-1, keepdims=True) * amplitudes
    z = torch.sqrt(2 / num_rff_features) * torch.cos(torch.matmul(hidden / kernel_param, w.T) + b)
    return z


def amplitude_frequency_and_probs(d):
    if d >= 4096:
        upper = 200
    elif d >= 2048:
        upper = 150
    elif d >= 1024:
        upper = 120
    else:
        upper = 100
    x = np.linspace(1e-12, upper, 10000)
    p = compute_prob(d, x)
    return x, p


def compute_prob(d, x_range):
    prob = np.array(
        [mpmath.besselk((d - 1) / 2, x) * mpmath.power(x, (d - 1) / 2) for x in x_range],
    )
    normalized_prob = prob / prob.sum()
    return normalized_prob
