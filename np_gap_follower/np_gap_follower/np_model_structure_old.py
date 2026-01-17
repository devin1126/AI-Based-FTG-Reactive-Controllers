import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence

# Attention Mechanism Block
class FeatureSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.ln(x + attn_out)

# General Multi-Layer Perceptron (MLP) Block
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
      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    super(MLP, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.n_hidden_layers = n_hidden_layers

    self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    self.activation = activation

    self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias).to(device)
    self.linears = nn.ModuleList(
        [
            nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias)
            for _ in range(self.n_hidden_layers - 1)
        ]
    ).to(device)

    self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias).to(device)
    self.aggregate_step = aggregate_step
    if self.aggregate_step:
      self.num_latents = num_latents
      self.latent_hidden = int((hidden_size+num_latents)/2)
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

    else:
      return out
    


# PI-AttNP Implementation
class AttLNP(nn.Module):

    def __init__(self, control_dim, gap_dim=128, vel_embed_dim=32, R_dim=128, num_heads=4, enable_ode_model=True, device='cpu'):
        super().__init__()
        self.R_dim = R_dim

        self.combined_encoder = MLP(control_dim + gap_dim, R_dim, device=device)
        self.self_attn = FeatureSelfAttention(R_dim, num_heads)

        self.query_proj = nn.Linear(gap_dim, R_dim)
        self.enable_ode_model = enable_ode_model
        if self.enable_ode_model:
           self.decoder = MLP(2*R_dim + control_dim + gap_dim, control_dim * 2, device=device)
        else:
           self.decoder = MLP(2*R_dim + gap_dim, control_dim * 2, device=device)
        
        self.latent_encoder = MLP(gap_dim + control_dim, R_dim, aggregate_step=True, num_latents=R_dim, device=device)
        self.device = device

    def latent_path(self, context_x, context_y):
        combined = torch.cat([context_y, context_x], dim=-1)
        mu, log_sigma = self.latent_encoder(combined)
        sigma = 0.9 * torch.sigmoid(log_sigma) + 0.1
        return Normal(mu, sigma)

    def forward(self, query, pred_control=torch.tensor([]), target_y=torch.tensor([]), is_testing=False):

        # Parsing data from dataset query
        (context_x, context_y), target_x = query
        num_targets = target_x.shape[1]

        prior = self.latent_path(context_x, context_y)

        if is_testing:
            z = prior.rsample().unsqueeze(1).repeat(1, num_targets, 1)
        else:
            posterior = self.latent_path(target_x, target_y)
            z = posterior.rsample().unsqueeze(1).repeat(1, num_targets, 1)

        # Self-attention + cross-attention computations
        combined = torch.cat([context_y, context_x], dim=-1)
        rep = self.combined_encoder(combined)
        rep = self.self_attn(rep)

        k = v = rep
        q = self.query_proj(target_x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.R_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        deterministic_rep = torch.matmul(attn_weights, v)

        # Combining deterministic representation with sampled latent variable
        rep = torch.cat([deterministic_rep, z], dim=-1)
        
        # Computing predictive distribution over 'target_y'
        if self.enable_ode_model:
           decoder_in = torch.cat([rep, target_x, pred_control], dim=-1)
        else:
           decoder_in = torch.cat([rep, target_x], dim=-1)
        decoder_out = self.decoder(decoder_in)

        mu, log_sigma = torch.chunk(decoder_out, 2, dim=-1)
        sigma = 0.9 * F.softplus(log_sigma) + 0.1

        dist = Independent(Normal(mu, sigma), 1)

        # Compute loss parameters and return predictions 
        if target_y.any():
            posterior = self.latent_path(target_x, target_y)
            kl = kl_divergence(posterior, prior).sum(-1, keepdim=True).repeat(1, num_targets)
            log_prob = dist.log_prob(target_y)
            return log_prob, kl, mu, sigma

        return mu, sigma