import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson


class PoissonTNNBlock(nn.Module):
    def __init__(self, num_obs, num_act, token_dim):
        super().__init__()

        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        self.encoder = nn.Sequential(
            nn.Linear(num_obs + 64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.lambda_head = nn.Sequential(
            nn.Linear(128, num_act),
            nn.Softplus()
        )

        self.policy_net = nn.Sequential(
            nn.Linear(num_act, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_act)
        )

        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, obs, token):
        token = self.token_proj(token)
        x = torch.cat([obs, token], dim=1)
        x = self.encoder(x)

        lam = self.lambda_head(x)
        lam = torch.nan_to_num(lam, nan=1.0, posinf=10.0, neginf=0.1)
        lam = lam.clamp(min=1e-3, max=10.0)

        poisson_sample = torch.poisson(lam)
        poisson_sample = poisson_sample.clamp(max=10)

        out = self.policy_net(poisson_sample)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        mu = torch.tanh(out)
        std = F.softplus(self.log_std) + 1e-3

        mu = torch.nan_to_num(mu, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=10.0, neginf=1e-3)

        return Normal(mu, std), lam
