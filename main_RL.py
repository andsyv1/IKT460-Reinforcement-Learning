import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from Es_agents import EnsemblePolicy


# --- Konfigurasjon ---
ENV_NAME = "InvertedPendulum-v4"
EPISODES = 1000
STEPS_PER_EPISODE = 1000
LEARNING_RATE = 4.5e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 20
TOKEN_DIM = 4

# --- Initiering ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Vi bruker EnsemblePolicy
policy = EnsemblePolicy(obs_dim, act_dim, TOKEN_DIM).to(device)

critic = nn.Sequential(
    nn.Linear(obs_dim, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1)
).to(device)

optimizer_policy = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
optimizer_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE)


def train():
    print("Starter PPO-treningsløkke...")

    for ep in range(EPISODES):
        obs, _ = env.reset()
        log_probs, rewards, values, states, actions = [], [], [], [], []
        total_reward = 0

        for _ in range(STEPS_PER_EPISODE):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Kall ensemble-policyen
            dist, _ = policy(obs_tensor)
            value = critic(obs_tensor)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            action_env = action.detach().cpu().numpy().reshape(-1)
            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            total_reward += reward

            states.append(obs_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            obs = next_obs
            if terminated or truncated:
                break

        returns, advantages = [], []
        G = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            G = r + GAMMA * G
            returns.insert(0, G)
            advantages.insert(0, G - v.item())

        returns = torch.tensor(returns, device=device).unsqueeze(1)
        advantages = torch.tensor(advantages, device=device).unsqueeze(1)
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)

        for _ in range(UPDATE_EPOCHS):
            new_dist, _ = policy(states)
            new_log_probs = new_dist.log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = torch.exp(new_log_probs - log_probs.detach().unsqueeze(1))

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (critic(states) - returns).pow(2).mean()

            optimizer_policy.zero_grad()
            optimizer_critic.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()

            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer_policy.step()
            optimizer_critic.step()

        print(f"Episode {ep + 1}/{EPISODES} - Reward: {total_reward:.2f}")

    print("Trening fullført!")

if __name__ == "__main__":
    train()
