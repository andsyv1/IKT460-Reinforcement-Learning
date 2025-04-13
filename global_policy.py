import torch
import torch.nn as nn
from VNN import EnhancedVNNBlock
from BerTNN import BernoulliTNNBlock
from PoisTNN import PoissonTNNBlock
from TokenLearner import TokenLearner  # Brukes for adaptive tokens

class GlobalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, token_dim=4):
        super().__init__()
        self.token_dim = token_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.token_learner = TokenLearner(obs_dim, act_dim, token_dim)

        self.poisson_agent = PoissonTNNBlock(obs_dim, act_dim, token_dim)
        self.bernoulli_agent = BernoulliTNNBlock(obs_dim, act_dim, token_dim)
        self.vnn_agent = EnhancedVNNBlock(obs_dim, act_dim, token_dim)

        # Dimensjonsberegning for z: combined_token + (obs + action + combined_token) + (3 * act_dim)
        z_dim = token_dim * 4 + obs_dim + 4 * act_dim


        self.fusion_net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, act_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs, action=None, reward=None):
        B = obs.size(0)

        if action is None:
            action = torch.zeros(B, self.act_dim, device=obs.device)
        if reward is None:
            reward = torch.zeros(B, 1, device=obs.device)

        # Hent ut de tre typene tokens
        combined_token, token_understanding, token_action = self.token_learner(obs, action, reward)

        # Spesifikke agenter får spesifikke tokens
        vnn_dist = self.vnn_agent(obs, token_understanding)
        pois_dist, lam = self.poisson_agent(obs, token_action)
        bern_dist, _ = self.bernoulli_agent(obs, token_action, lam)

        # Sammenslåing for global policy
        x = torch.cat([obs, action, combined_token], dim=1)
        y = torch.cat([pois_dist.mean, bern_dist.mean, vnn_dist.mean], dim=1)

        # Rens NaN og Inf i x og y
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        z = torch.cat([combined_token, x, y], dim=1)
        z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

        mu = torch.tanh(self.fusion_net(z))
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        std = torch.exp(self.log_std).expand_as(mu)

        dist = torch.distributions.Normal(mu, std)
        entropy = dist.entropy().sum(dim=1)

        return dist, entropy
