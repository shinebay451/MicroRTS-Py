import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
import wandb


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


class Agent(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        h, w, c = input_shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * h * w, 256)),
            nn.ReLU()
        )
        self.policy_head = layer_init(nn.Linear(256, n_actions), std=0.01)

    def forward(self, x):
        x = self.encoder(x)
        return self.policy_head(x)

    def get_action(self, x):
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)


def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)


def main():
    wandb.init(project="microrts-reinforce", name=f"run_{int(time.time())}", config={"lr": 1e-3, "gamma": 0.99, "episodes": 1000})
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.workerRushAI],
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([1.0]),
        cycle_maps=["maps/16x16/basesWorkers16x16A.xml"]
    )

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = Agent(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr)

    n_episodes = config.episodes
    gamma = config.gamma

    for episode in range(n_episodes):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        log_probs = []
        rewards = []
        done = False

        while not done:
            action, log_prob = agent.get_action(obs)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            log_probs.append(log_prob)
            rewards.append(reward)

        returns = compute_returns(rewards, gamma).to(device)
        loss = -(torch.stack(log_probs) * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        wandb.log({"episode": episode, "return": total_reward, "loss": loss.item()})
        print(f"Episode {episode}, Return: {total_reward:.2f}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
