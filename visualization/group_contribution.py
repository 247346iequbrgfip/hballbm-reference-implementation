import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_group_contribution(
    H,
    group_dims=(10, 10, 10),
):
    T, N, D = H.shape
    dI, dC, dG = group_dims
    assert dI + dC + dG == D

    A_I, A_C, A_G = [], [], []

    for t in range(T):
        H_t = H[t]
        H_t = H_t.clone().detach().requires_grad_(True)

        score = (H_t ** 2).sum()

        grad = torch.autograd.grad(score, H_t)[0]

        grad_I = grad[:, :dI]
        grad_C = grad[:, dI:dI + dC]
        grad_G = grad[:, dI + dC:]

        A_I.append(torch.sum(torch.abs(grad_I)).item())
        A_C.append(torch.sum(torch.abs(grad_C)).item())
        A_G.append(torch.sum(torch.abs(grad_G)).item())

    A_I = np.array(A_I)
    A_C = np.array(A_C)
    A_G = np.array(A_G)

    denom = A_I + A_C + A_G

    R_I = A_I / denom
    R_C = A_C / denom
    R_G = A_G / denom

    return R_I, R_C, R_G


def plot_group_contribution(R_I, R_C, R_G):
    plt.figure(figsize=(6, 4))
    plt.plot(R_I, label="Imaging (I)", linewidth=2)
    plt.plot(R_C, label="Clinical (C)", linewidth=2)
    plt.plot(R_G, label="Genomic (G)", linewidth=2)
    plt.xlabel("Time t")
    plt.ylabel("Relative Contribution")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
