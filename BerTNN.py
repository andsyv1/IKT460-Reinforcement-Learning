import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BernoulliTNNBlock(nn.Module):
    def __init__(self, num_obs, num_act, token_dim):
        super().__init__()

        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.ReLU()
        )

        self.encoder = nn.Sequential(
            nn.Linear(num_obs + 64 + num_act, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.prob_head = nn.Sequential(
            nn.Linear(128, num_act),
            nn.Sigmoid()
        )

        self.policy_net = nn.Sequential(
            nn.Linear(num_act, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, num_act)
        )

        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, obs, token, lam):
        token = self.token_proj(token)
        x = torch.cat([obs, token, lam], dim=1)
        x = self.encoder(x)

        probs = self.prob_head(x)
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        probs = probs.clamp(1e-5, 1 - 1e-5)

        bern_sample = torch.bernoulli(probs)
        bern_sample = torch.nan_to_num(bern_sample, nan=0.0)

        out = self.policy_net(bern_sample)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        mu = torch.tanh(out)
        std = F.softplus(self.log_std) + 1e-3

        mu = torch.nan_to_num(mu, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=10.0, neginf=1e-3)

        return Normal(mu, std), probs
