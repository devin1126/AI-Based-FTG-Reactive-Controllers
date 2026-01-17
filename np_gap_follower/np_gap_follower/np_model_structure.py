import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence


# -----------------------------
# Attention Mechanism Block
# -----------------------------
class FeatureSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.ln(x + attn_out)


# -----------------------------
# General Multi-Layer Perceptron (MLP) Block
# -----------------------------
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        n_hidden_layers=1,
        activation=nn.ReLU(),
        is_bias=True,
        dropout=0,
        aggregate_step=False,
        num_latents=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.activation = activation

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias).to(device)
        self.linears = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias) for _ in range(self.n_hidden_layers - 1)]
        ).to(device)

        self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias).to(device)

        self.aggregate_step = aggregate_step
        if self.aggregate_step:
            self.num_latents = num_latents
            self.latent_hidden = int((hidden_size + num_latents) / 2)
            self.penultimate_layer = nn.Linear(output_size, self.latent_hidden, bias=is_bias).to(device)
            self.mu_layer = nn.Linear(self.latent_hidden, num_latents).to(device)
            self.log_sigma_layer = nn.Linear(self.latent_hidden, num_latents).to(device)

    def forward(self, x):
        # Ensure input tensor is on the same device as the model
        x = x.to(self.to_hidden.weight.device)

        out = self.to_hidden(x)
        out = self.activation(out)
        x = self.dropout(out)

        for linear in self.linears:
            out = linear(x)
            out = self.activation(out)
            out = self.dropout(out)
            x = out

        out = self.out(out)

        # Add-in for latent encoder steps
        if self.aggregate_step:
            out = torch.mean(out, dim=1)
            out = self.penultimate_layer(out)
            mu = self.mu_layer(out)
            log_sigma = self.log_sigma_layer(out)
            return mu, log_sigma

        return out


# -----------------------------
# PI-AttNP Implementation w/ Velocity Embedding
# -----------------------------
class AttLNP(nn.Module):
    """
    Updated AttLNP to incorporate a velocity embedding MLP.

    Expected data layout (recommended):
      - context_x, target_x each contain: [gap_dim, v, w]  -> total dim = gap_dim + 2
      - context_y, target_y contain: steering angle (control_dim=1)

    Then inside the model:
      - (v,w) -> vel_encoder -> vel_emb_dim
      - x_emb = [gap_dim, vel_emb_dim] -> total x_dim = gap_dim + vel_emb_dim
    """

    def __init__(
        self,
        control_dim: int,
        gap_dim: int = 128,
        vel_embed_dim: int = 32,
        R_dim: int = 128,
        num_heads: int = 4,
        enable_ode_model: bool = True,
        device: str = "cpu",
        vel_max: float = 5.0
    ):
        super().__init__()
        self.R_dim = R_dim
        self.gap_dim = gap_dim
        self.vel_embed_dim = vel_embed_dim
        self.control_dim = control_dim

        self.enable_ode_model = enable_ode_model
        self.device = device

        # ---- Velocity encoder (2 -> vel_embed_dim) ----
        self.vel_max = vel_max
        self.vel_encoder = MLP(
            input_size=2,
            output_size=vel_embed_dim,
            hidden_size=32,
            n_hidden_layers=1,
            activation=nn.ReLU(),
            device=device,
        )

        # This is the "effective" x dimension used everywhere inside the NP
        self.x_emb_dim = gap_dim + vel_embed_dim

        # ---- Encoders / attention ----
        # combined = [y (control_dim), x_emb (gap+vel_emb)]
        self.combined_encoder = MLP(control_dim + self.x_emb_dim, R_dim, device=device)
        self.self_attn = FeatureSelfAttention(R_dim, num_heads)

        # Query projection takes target_x_emb
        self.query_proj = nn.Linear(self.x_emb_dim, R_dim)

        # Decoder conditions on [rep (2R), target_x_emb, (optional) pred_control]
        if self.enable_ode_model:
            self.decoder = MLP(2 * R_dim + self.x_emb_dim + control_dim, control_dim * 2, device=device)
        else:
            self.decoder = MLP(2 * R_dim + self.x_emb_dim, control_dim * 2, device=device)

        # Latent path encoder aggregates over (control + x_emb)
        self.latent_encoder = MLP(
            self.x_emb_dim + control_dim,
            R_dim,
            aggregate_step=True,
            num_latents=R_dim,
            device=device,
        )

    def _embed_x(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: [..., gap_dim + 2] = [gap, v, w]
        returns x_emb: [..., gap_dim + vel_embed_dim]
        """
        x_raw = x_raw.to(next(self.parameters()).device)

        if x_raw.shape[-1] != self.gap_dim + 2:
            raise ValueError(
                f"AttLNP expected x_raw last dim = gap_dim+2 = {self.gap_dim+2}, "
                f"but got {x_raw.shape[-1]}. Ensure context_x/target_x contain [gap, v, w] only."
            )

        gap = x_raw[:, :, :self.gap_dim]
        vel = x_raw[:, :, self.gap_dim:] / self.vel_max  # normalize by max vel for stability

        vel_emb = self.vel_encoder(vel)  # [..., vel_embed_dim]
        x_emb = torch.cat([gap, vel_emb], dim=-1)
        return x_emb

    # -----------------------------
    # Latent path
    # -----------------------------
    def latent_path(self, context_x: torch.Tensor, context_y: torch.Tensor) -> Normal:
        combined = torch.cat([context_y, context_x], dim=-1)  # [B, C, control + x_emb]
        mu, log_sigma = self.latent_encoder(combined)
        sigma = 0.9 * torch.sigmoid(log_sigma) + 0.1
        return Normal(mu, sigma)

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, query, pred_control=torch.tensor([]), target_y=torch.tensor([]), is_testing: bool = False):
        (context_x_raw, context_y), target_x_raw = query
        num_targets = target_x_raw.shape[1]

        # Embed x's
        context_x = self._embed_x(context_x_raw)   # [B, C, x_emb_dim]
        target_x = self._embed_x(target_x_raw)     # [B, T, x_emb_dim]

        # Prior / posterior over z
        prior = self.latent_path(context_x, context_y)

        if is_testing:
            z = prior.rsample().unsqueeze(1).repeat(1, num_targets, 1)
        else:
            posterior = self.latent_path(target_x, target_y)
            z = posterior.rsample().unsqueeze(1).repeat(1, num_targets, 1)

        # Self-attention + cross-attention
        combined = torch.cat([context_y, context_x], dim=-1)  # [B, C, control + x_emb]
        rep = self.combined_encoder(combined)                 # [B, C, R_dim]
        rep = self.self_attn(rep)                             # [B, C, R_dim]

        k = v = rep
        q = self.query_proj(target_x)                         # [B, T, R_dim]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.R_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        deterministic_rep = torch.matmul(attn_weights, v)     # [B, T, R_dim]

        # Combine deterministic rep and sampled z
        rep = torch.cat([deterministic_rep, z], dim=-1)       # [B, T, 2R_dim]

        # Decoder
        if self.enable_ode_model:
            if pred_control.numel() == 0:
                raise ValueError("enable_ode_model=True but pred_control was not provided.")
            pred_control = pred_control.to(rep.device)
            decoder_in = torch.cat([rep, target_x, pred_control], dim=-1)
        else:
            decoder_in = torch.cat([rep, target_x], dim=-1)

        decoder_out = self.decoder(decoder_in)
        mu, log_sigma = torch.chunk(decoder_out, 2, dim=-1)
        sigma = 0.9 * F.softplus(log_sigma) + 0.1
        dist = Independent(Normal(mu, sigma), 1)

        # Training outputs
        if target_y.numel() > 0 and target_y.any():
            posterior = self.latent_path(target_x, target_y)
            kl = kl_divergence(posterior, prior).sum(-1, keepdim=True).repeat(1, num_targets)
            log_prob = dist.log_prob(target_y)
            return log_prob, kl, mu, sigma

        # Inference outputs
        return mu, sigma
