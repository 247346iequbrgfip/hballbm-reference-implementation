import torch
import torch.nn as nn


class HBVectorField(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, h: torch.Tensor, t_scalar: float) -> torch.Tensor:
        t = torch.full((h.shape[0], 1), float(t_scalar), device=h.device, dtype=h.dtype)
        return self.f(torch.cat([h, t], dim=1))


def heavy_ball_step(
    h: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    t: float,
    field: nn.Module,
    gamma: float,
):
    a = field(h, t) - gamma * v
    v_next = v + dt * a
    h_next = h + dt * v_next
    return h_next, v_next
