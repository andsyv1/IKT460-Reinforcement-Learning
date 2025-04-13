import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class EnhancedVNNBlock(nn.Module):
    def __init__(self, num_obs, num_act, token_dim=4):
        super(EnhancedVNNBlock, self).__init__()

        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.Tanh(),
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.Tanh(),
            nn.LayerNorm(token_dim)
        )

        self.fc1 = nn.Linear(num_obs + token_dim, 32)
        self.norm1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 64)
        self.norm2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 128)
        self.norm3 = nn.LayerNorm(128)
        self.out = nn.Linear(128, num_act)

        self.variance_fc1_act = nn.Sequential(
            nn.Linear(32, 64), nn.LeakyReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128), nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 164), nn.LeakyReLU(),
            nn.LayerNorm(164),
            nn.Linear(164, num_act)
        )

        self.variance_fc2_act = nn.Sequential(
            nn.Linear(64, 128), nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 164), nn.LeakyReLU(),
            nn.LayerNorm(164),
            nn.Linear(164, 186), nn.LeakyReLU(),
            nn.LayerNorm(186),
            nn.Linear(186, num_act)
        )

        self.variance_fc3_act = nn.Sequential(
            nn.Linear(128, 164), nn.LeakyReLU(),
            nn.LayerNorm(164),
            nn.Linear(164, 186), nn.LeakyReLU(),
            nn.LayerNorm(186),
            nn.Linear(186, num_act)
        )

        self.act_net1 = nn.Linear(num_act * 4, 310)
        self.norm_act1 = nn.LayerNorm(310)
        self.dropout1 = nn.Dropout(0.1)
        self.act_net2 = nn.Linear(310, 155)
        self.norm_act2 = nn.LayerNorm(155)
        self.dropout2 = nn.Dropout(0.1)
        self.act_net3 = nn.Linear(155, num_act)

        self.log_std = nn.Parameter(torch.zeros(num_act))

    def forward(self, obs, token=None):
        if token is not None:
            token_processed = self.token_proj(token)
            obs = torch.cat([obs, token_processed], dim=1)

        x1 = F.leaky_relu(self.norm1(self.fc1(obs)))
        x2 = F.leaky_relu(self.norm2(self.fc2(x1)))
        x3 = F.leaky_relu(self.norm3(self.fc3(x2)))
        x_main = self.out(x3)

        batch_size = obs.shape[0]

        if batch_size > 1:
            var1 = torch.var(x1, dim=0, keepdim=True).expand(batch_size, -1)
            var2 = torch.var(x2, dim=0, keepdim=True).expand(batch_size, -1)
            var3 = torch.var(x3, dim=0, keepdim=True).expand(batch_size, -1)

            var1 = torch.nan_to_num(var1, nan=0.0)
            var2 = torch.nan_to_num(var2, nan=0.0)
            var3 = torch.nan_to_num(var3, nan=0.0)

            out1 = self.variance_fc1_act(var1)
            out2 = self.variance_fc2_act(var2)
            out3 = self.variance_fc3_act(var3)

            variance_features = torch.cat([out1, out2, out3], dim=1)

            combined = torch.cat([x_main, variance_features], dim=1)

            a = F.leaky_relu(self.norm_act1(self.act_net1(combined)))
            a = self.dropout1(a)
            a = F.leaky_relu(self.norm_act2(self.act_net2(a)))
            a = self.dropout2(a)
            mu = torch.tanh(self.act_net3(a))
        else:
            mu = torch.tanh(x_main)

        std = F.softplus(self.log_std) + 1e-3

        mu = torch.nan_to_num(mu, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=10.0, neginf=1e-3)

        return Normal(mu, std)



