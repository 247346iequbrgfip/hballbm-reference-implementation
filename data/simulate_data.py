import numpy as np
import torch


def make_synthetic_multimodal_dataset(
    n: int = 400,
    d_img: int = 64,
    d_cli: int = 16,
    d_gen: int = 16,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    x_img = rng.normal(size=(n, d_img)).astype(np.float32)
    x_cli = rng.normal(size=(n, d_cli)).astype(np.float32)
    x_gen = rng.normal(size=(n, d_gen)).astype(np.float32)

    # Generic synthetic signal
    s = (
        0.7 * x_img[:, 0]
        - 0.3 * x_img[:, 3]
        + 0.6 * x_cli[:, 1]
        - 0.4 * x_cli[:, 4]
        + 0.5 * x_gen[:, 2]
        + 0.2 * np.sin(x_gen[:, 5])
        + 0.25 * rng.normal(size=n).astype(np.float32)
    )
    y = (s > 0.0).astype(np.int64)

    return (
        torch.from_numpy(x_img),
        torch.from_numpy(x_cli),
        torch.from_numpy(x_gen),
        torch.from_numpy(y),
    )
