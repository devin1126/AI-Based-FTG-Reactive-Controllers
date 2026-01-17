import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence


class ResidualGatedMLPBlock(nn.Module):
    """Pre-norm residual gated MLP block (GEGLU-ish) for stability."""
    def __init__(self, hidden_size: int, dropout: float = 0.0, resid_scale: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.resid_scale = resid_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, g = self.fc1(h).chunk(2, dim=-1)
        h = a * F.gelu(g)
        h = self.fc2(self.drop(h))
        return x + self.resid_scale * h


class ResGatedMLP(nn.Module):
    """
    A stabilized MLP: input projection + (pre-norm residual blocks) + output head.
    Works for both [B,D] and [B,T,D] by operating on the last dimension.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
        resid_scale: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_size)
        self.in_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([
            ResidualGatedMLPBlock(hidden_size, dropout=dropout, resid_scale=resid_scale)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.in_norm(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return self.out(h)


class FeatureSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.05):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        return x + self.drop(attn_out)



class AttLNP(nn.Module):
    def __init__(
        self,
        control_dim: int,
        gap_dim: int = 128,
        R_dim: int = 128,
        num_heads: int = 4,
        enable_ode_model: bool = True,
        hidden_size: int = 128,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dropout: float = 0.05,
        resid_scale: float = 0.1,
        min_std: float = 1e-3,
        max_std: float = 1.0,
        z_scale: float = 1.0,  # can set <1.0 early in training if needed
        device: str = "cpu",
    ):
        super().__init__()
        self.R_dim = R_dim
        self.enable_ode_model = enable_ode_model
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.z_scale = float(z_scale)

        # Deterministic encoder: (y,x)->R
        self.combined_encoder = ResGatedMLP(
            input_dim=control_dim + gap_dim,
            output_dim=R_dim,
            hidden_size=hidden_size,
            num_layers=enc_layers,
            dropout=dropout,
            resid_scale=resid_scale,
        )

        self.self_attn = FeatureSelfAttention(R_dim, num_heads=num_heads, dropout=dropout)

        # Cross-attention projections
        self.query_proj = nn.Linear(gap_dim, R_dim)
        self.key_norm = nn.LayerNorm(R_dim)
        self.query_norm = nn.LayerNorm(R_dim)

        # Latent encoder: aggregate over set -> (mu, std)
        # We implement aggregation explicitly (mean pooling) for stability.
        self.latent_encoder = ResGatedMLP(
            input_dim=control_dim + gap_dim,
            output_dim=2 * R_dim,  # mu and raw_std
            hidden_size=hidden_size,
            num_layers=enc_layers,
            dropout=dropout,
            resid_scale=resid_scale,
        )

        # Decoder: input is [det_rep, z, x, (pred_control)]
        dec_in_dim = 2 * R_dim + gap_dim + (control_dim if enable_ode_model else 0)
        self.decoder = ResGatedMLP(
            input_dim=dec_in_dim,
            output_dim=2 * control_dim,  # mu and raw_std
            hidden_size=hidden_size,
            num_layers=dec_layers,
            dropout=dropout,
            resid_scale=resid_scale,
        )

        self.to(device)

    def _make_normal(self, mu: torch.Tensor, raw_std: torch.Tensor) -> Normal:
        std = F.softplus(raw_std) + self.min_std
        std = torch.clamp(std, min=self.min_std, max=self.max_std)
        return Normal(mu, std)

    def latent_path(self, context_x: torch.Tensor, context_y: torch.Tensor) -> Normal:
        # combined: [B, Nc, gap+control]
        combined = torch.cat([context_y, context_x], dim=-1)
        # encode each element, then aggregate
        stats = self.latent_encoder(combined)          # [B, Nc, 2R]
        stats = stats.mean(dim=1)                      # [B, 2R]
        mu, raw_std = stats.chunk(2, dim=-1)           # each [B, R]
        return self._make_normal(mu, raw_std)

    def forward(self, query, pred_control: torch.Tensor = None, target_y: torch.Tensor = None, is_testing: bool = False):
        (context_x, context_y), target_x = query   # context_x: [B,Nc,gap], context_y: [B,Nc,ctrl], target_x: [B,Nt,gap]
        B, Nt, _ = target_x.shape

        prior = self.latent_path(context_x, context_y)

        if (not is_testing) and (target_y is not None):
            posterior = self.latent_path(target_x, target_y)
            z = posterior.rsample() * self.z_scale
            kl = kl_divergence(posterior, prior).sum(-1, keepdim=True)  # [B,1]
        else:
            z = prior.rsample() * self.z_scale
            kl = None

        z = z.unsqueeze(1).expand(B, Nt, self.R_dim)  # [B,Nt,R]

        # Deterministic path
        combined = torch.cat([context_y, context_x], dim=-1)      # [B,Nc,ctrl+gap]
        rep = self.combined_encoder(combined)                    # [B,Nc,R]
        rep = self.self_attn(rep)

        k = self.key_norm(rep)
        v = rep
        q = self.query_norm(self.query_proj(target_x))           # [B,Nt,R]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.R_dim)  # [B,Nt,Nc]
        attn_weights = F.softmax(attn_scores, dim=-1)
        deterministic_rep = torch.matmul(attn_weights, v)        # [B,Nt,R]

        rep = torch.cat([deterministic_rep, z], dim=-1)          # [B,Nt,2R]

        if self.enable_ode_model:
            if pred_control is None:
                raise ValueError("enable_ode_model=True requires pred_control.")
            # ensure pred_control is [B,Nt,control_dim]
            if pred_control.dim() == 2:
                pred_control = pred_control.unsqueeze(1).expand(B, Nt, -1)
            decoder_in = torch.cat([rep, target_x, pred_control], dim=-1)
        else:
            decoder_in = torch.cat([rep, target_x], dim=-1)

        decoder_out = self.decoder(decoder_in)                   # [B,Nt,2*control_dim]
        mu, raw_std = decoder_out.chunk(2, dim=-1)
        dist = Independent(self._make_normal(mu, raw_std), 1)

        if target_y is not None:
            # log_prob: [B,Nt]
            log_prob = dist.log_prob(target_y)
            if kl is None:
                # testing path with provided target (rare)
                posterior = self.latent_path(target_x, target_y)
                kl = kl_divergence(posterior, prior).sum(-1, keepdim=True)

            kl = kl.expand(B, Nt)  # [B,Nt]
            return log_prob, kl, mu, dist.base_dist.scale  # returning sigma for your metrics

        return mu, dist.base_dist.scale
