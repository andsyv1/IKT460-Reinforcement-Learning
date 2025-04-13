from BerTNN import BernoulliTNNBlock
from PoisTNN import PoissonTNNBlock
from VNN import EnhancedVNNBlock
import torch
import torch.nn as nn

class EnsemblePolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, token_dim=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.token_dim = token_dim

        # Tre del-policyer
        self.bern_policy = BernoulliTNNBlock(obs_dim, act_dim, token_dim)
        self.poiss_policy = PoissonTNNBlock(obs_dim, act_dim, token_dim)
        self.vnn_policy  = EnhancedVNNBlock(obs_dim, act_dim, token_dim)

        # Gating-nett for å bestemme vekter til de tre policyene
        self.gate_net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # output: [w_bern, w_poiss, w_vnn]
        )

    def forward(self, obs):
        """
        Returnerer (dist, extra_info) slik at du kan bruke:
            dist.sample(), dist.log_prob(...), dist.mean, dist.stddev, etc.
        """

        batch_size = obs.size(0)

        # Kall hver sub-policy med en dummy-token (evt. lam) for å få en Normal-fordeling
        dummy_token = torch.zeros(batch_size, self.token_dim, device=obs.device)
        lam_dummy   = torch.ones(batch_size, self.act_dim, device=obs.device)

        # Bernoulli sub-policy => Normal
        dist_bern, _ = self.bern_policy(obs, dummy_token, lam_dummy)

        # Poisson sub-policy => Normal
        dist_poiss, _ = self.poiss_policy(obs, dummy_token)

        # VNN sub-policy => Normal
        dist_vnn = self.vnn_policy(obs, dummy_token)

        # Tre sub-fordelinger: henter mean og std
        mu_bern, std_bern   = dist_bern.mean, dist_bern.stddev
        mu_poiss, std_poiss = dist_poiss.mean, dist_poiss.stddev
        mu_vnn, std_vnn     = dist_vnn.mean, dist_vnn.stddev

        # Hent gating-vekter
        gate_logits = self.gate_net(obs)  # (batch_size, 3)
        weights_raw = torch.softmax(gate_logits, dim=-1)  # w_bern, w_poiss, w_vnn; sum=1

        # Ekspander til shape [batch_size, 3, act_dim]
        w = weights_raw.unsqueeze(-1).expand(batch_size, 3, self.act_dim)

        # Samle mus og stds i en [batch_size, 3, act_dim]
        mus = torch.stack([mu_bern, mu_poiss, mu_vnn], dim=1)
        stds = torch.stack([std_bern, std_poiss, std_vnn], dim=1)

        # 1) Kombinert mean
        #    mu = sum_i [ w_i * mu_i ]
        mu_ensemble = (w * mus).sum(dim=1)  # [batch_size, act_dim]

        # 2) Kombinert varianse (simplifisert formel)
        #    var = sum_i [ w_i * (std_i^2 + mu_i^2) ] - mu_ensemble^2
        var_ensemble = (w * (stds**2 + mus**2)).sum(dim=1) - mu_ensemble**2
        var_ensemble = torch.clamp(var_ensemble, min=1e-5)  # unngå negative

        std_ensemble = torch.sqrt(var_ensemble)

        # Endelig dist
        dist = torch.distributions.Normal(mu_ensemble, std_ensemble)

        return dist, {"weights": weights_raw}
