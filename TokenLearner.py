import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    def __init__(self, obs_dim, act_dim, token_dim):
        super().__init__()

        input_dim = obs_dim + act_dim + 1  # 1 for reward

        self.net1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, token_dim)
        )

        self.net2 = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, token_dim)
        )

        self.net3 = nn.Sequential(
            nn.Linear(act_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, token_dim),
            #nn.LayerNorm(256),
            #nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            #nn.LayerNorm(128),
            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.LayerNorm(64),
            #nn.Linear(64, token_dim)
        )

        self.net4 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, token_dim)
        )

        # Optional fusion layer:
        self.fusion_understanding = nn.Linear(token_dim * 2, token_dim)
        self.fusion_action = nn.Linear(token_dim * 2, token_dim)

    def forward(self, obs, action, reward):
        x = torch.cat([obs, action, reward], dim=1)

        token1 = self.net1(x)
        token2 = self.net2(obs)
        token3 = self.net3(action)
        token4 = self.net4(x)

        #combined = torch.cat([token1, token2, token3, token4], dim=1)
        combined1 = torch.cat([token1, token2], dim=1)
        combined2 = torch.cat([token3, token4], dim=1)

        token_understanding = self.fusion_understanding(combined1)
        token_action = self.fusion_action(combined2)

        combined = torch.cat([token_understanding, token_action], dim=1)

        return combined, token_understanding, token_action
