import torch
import torch.nn as nn
from models.encoders import ModalityEncoder
from models.hb_dynamics import HBVectorField, heavy_ball_step


class MultimodalHBModel(nn.Module):
    def __init__(
        self,
        d_img: int,
        d_cli: int,
        d_gen: int,
        latent_dim: int = 64,
        n_classes: int = 2,
        gamma: float = 0.8,
    ):
        super().__init__()
        self.gamma = float(gamma)

        self.enc_img = ModalityEncoder(d_img, latent_dim)
        self.enc_cli = ModalityEncoder(d_cli, latent_dim)
        self.enc_gen = ModalityEncoder(d_gen, latent_dim)

        self.fuse = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.field = HBVectorField(latent_dim=latent_dim, hidden=128)

        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, n_classes),
        )

    def _init_state(self, h0: torch.Tensor):
        v0 = torch.zeros_like(h0)
        return h0, v0

    def evolve(self, h0: torch.Tensor, timesteps: torch.Tensor, retain_grads: bool):
        T = timesteps.numel()
        H = []

        h, v = self._init_state(h0)

        for k in range(T):
            if retain_grads:
                h.retain_grad()
            H.append(h)

            if k < T - 1:
                t = float(timesteps[k].item())
                dt = float((timesteps[k + 1] - timesteps[k]).item())
                h, v = heavy_ball_step(h, v, dt, t, self.field, self.gamma)

        return torch.stack(H, dim=0)

    def forward(self, x_img, x_cli, x_gen, timesteps, retain_trajectory_grads: bool = False):
        z_img = self.enc_img(x_img)
        z_cli = self.enc_cli(x_cli)
        z_gen = self.enc_gen(x_gen)

        h0 = self.fuse(torch.cat([z_img, z_cli, z_gen], dim=1))
        H = self.evolve(h0, timesteps, retain_grads=retain_trajectory_grads)
        logits = self.head(H[-1])
        return logits, H
