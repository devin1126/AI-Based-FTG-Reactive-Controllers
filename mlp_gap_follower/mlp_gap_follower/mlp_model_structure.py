import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# --------------------------------------------------
# Residual gated block (unchanged)
# --------------------------------------------------
class ResidualGatedMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0, resid_scale: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.resid_scale = resid_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, g = self.fc1(h).chunk(2, dim=-1)
        h = a * F.gelu(g)
        h = self.fc2(self.dropout(h))
        return x + self.resid_scale * h


# --------------------------------------------------
# Residual MLP Policy w/ Velocity Embedding
# --------------------------------------------------
class ResMLPPolicy(nn.Module):
    """
    Residual MLP baseline with velocity embedding MLP.

    Expected input:
      x_raw = [gap_dim, v, w]    -> shape (B, gap_dim + 2)

    Inside model:
      (v, w) -> vel_encoder -> vel_embed_dim
      x_emb = [gap, vel_emb]
    """

    def __init__(
        self,
        gap_dim: int = 128,
        vel_embed_dim: int = 32,
        hidden_size: int = 128,
        num_layers: int = 4,
        output_size: int = 1,
        max_steer: float = 0.6981,
        dropout: float = 0.05,
        resid_scale: float = 0.1,
        min_std: float = 1e-3,
        max_std: float = 0.5,
        vel_max: float = 5.0,   # max linear / angular velocity magnitude
    ):
        super().__init__()

        self.gap_dim = gap_dim
        self.vel_embed_dim = vel_embed_dim
        self.vel_max = float(vel_max)

        self.max_steer = float(max_steer)
        self.min_std = float(min_std)
        self.max_std = float(max_std)

        # -----------------------------
        # Velocity embedding MLP (2 → vel_embed_dim)
        # -----------------------------
        self.vel_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, vel_embed_dim),
        )

        # Effective input dimension after embedding
        input_dim = gap_dim + vel_embed_dim

        # -----------------------------
        # Main residual MLP trunk
        # -----------------------------
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList([
            ResidualGatedMLPBlock(
                hidden_size,
                dropout=dropout,
                resid_scale=resid_scale
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)

        # Output heads
        self.mean_head = nn.Linear(hidden_size, output_size)
        self.log_std_head = nn.Linear(hidden_size, output_size)

        self._init_weights()

    # --------------------------------------------------
    # Weight init (unchanged but important)
    # --------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, math.log(0.05))

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x_raw: torch.Tensor, is_testing: bool = False):
        """
        x_raw: (B, gap_dim + 2) = [gap, v, w]
        """

        if x_raw.shape[-1] != self.gap_dim + 2:
            raise ValueError(
                f"Expected input dim {self.gap_dim + 2}, got {x_raw.shape[-1]}"
            )

        # -----------------------------
        # Split inputs
        # -----------------------------
        gap = x_raw[:, :self.gap_dim]
        vel = x_raw[:, self.gap_dim:]  # (B, 2) = [v, w]

        # Normalize velocities (NO in-place ops)
        vel = vel / self.vel_max

        # Velocity embedding
        vel_emb = self.vel_encoder(vel)

        # Concatenate geometry + velocity embedding
        x = torch.cat([gap, vel_emb], dim=-1)

        # -----------------------------
        # Residual MLP
        # -----------------------------
        h = self.input_proj(x)
        h = self.input_norm(h)

        for blk in self.blocks:
            h = blk(h)

        h = self.final_norm(h)

        # Mean
        mu = torch.tanh(self.mean_head(h)) * self.max_steer

        # Std
        raw = self.log_std_head(h)
        std = F.softplus(raw) + self.min_std
        std = torch.clamp(std, self.min_std, self.max_std)

        if not is_testing:
            dist = Normal(mu, std)
            return dist, mu, std

        return mu, std