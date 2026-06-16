"""
Models for Causal World Modelling in Dot World.
  - Encoder  f_θ :  observation  →  latent z
  - Forward  g_ϕ :  (z_t, a_t)  →  ẑ_{t+1}
  - Inverse  h_ψ :  (z_t, z_{t+1})  →  â_t
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────
#  Encoder
# ──────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    (3, H, W) image  →  (latent_dim,) vector in [-1, 1].

    Three stride-2 convolutions.
    """

    def __init__(self, obs_channels: int = 3, latent_dim: int = 64, image_size: int = 64):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {image_size}.")
        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        spatial = image_size // 16
        self.fc = nn.Linear(128 * spatial * spatial, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(start_dim=1)
        return torch.tanh(self.fc(h))


# ──────────────────────────────────────────────────────────────────
#  Forward dynamics model
# ──────────────────────────────────────────────────────────────────

class ForwardModel(nn.Module):
    """
    Predicts ẑ_{t+1} = g_ϕ(z_t, a_t).

    2-layer MLP. tanh activation.
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)  — current embedding.
            a: (B, action_dim)  — action taken.
        Returns:
            (B, latent_dim) predicted next embedding, in [-1, 1].
        """
        return torch.tanh(self.net(torch.cat([z, a], dim=-1)))


# ──────────────────────────────────────────────────────────────────
#  Inverse dynamics model
# ──────────────────────────────────────────────────────────────────

class InverseModel(nn.Module):
    """
    Predicts â_t = h_ψ(z_t, z_{t+1}).

    2-layer MLP. No output activation.
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:      (B, latent_dim) — current embedding.
            z_next: (B, latent_dim) — next embedding.
        Returns:
            (B, action_dim) predicted action (unbounded).
        """
        return self.net(torch.cat([z, z_next], dim=-1))


# ──────────────────────────────────────────────────────────────────
#  Probe decoder
# ──────────────────────────────────────────────────────────────────

class CNNDecoder(nn.Module):
    """
    (latent_dim,) vector  →  (obs_channels, H, W) image in [0, 1].

    Architectural inverse of CNNEncoder.
    """

    def __init__(self, obs_channels: int = 3, latent_dim: int = 64, image_size: int = 64):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {image_size}.")
        self.spatial = image_size // 16
        self.fc = nn.Linear(latent_dim, 128 * self.spatial * self.spatial)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) — embedding to reconstruct from.
        Returns:
            (B, obs_channels, H, W) reconstruction in [0, 1].
        """
        h = self.fc(z).view(-1, 128, self.spatial, self.spatial)
        return torch.sigmoid(self.deconv(h))