# http://proceedings.mlr.press/v97/han19a/han19a.pdf


import argparse
import os
import random
import subprocess
import time
from distutils.util import strtobool
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import (VecEnvWrapper, VecMonitor,
                                              VecVideoRecorder)
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total-timesteps', type=int, default=1000000)
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--wandb-project-name', type=str, default="gym-microrts")
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--reward-weight', type=float, nargs='+', default=[10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--num-bot-envs', type=int, default=0)
    parser.add_argument('--num-selfplay-envs', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--train-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"])
    parser.add_argument('--eval-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"])
    args = parser.parse_args()
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    return args


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class Agent(nn.Module):
    def __init__(self, envs, mapsize=16 * 16):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        h, w, c = envs.observation_space.shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, 78, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
        )
        self.register_buffer("mask_value", torch.tensor(-1e8))

    def get_action_and_logprob(self, x, invalid_action_masks, envs):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        grid_logits = logits.reshape(-1, envs.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envs.action_plane_space.nvec.tolist(), dim=1)
        invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
        split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=1)
        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
            for logits, iam in zip(split_logits, split_invalid_action_masks)
        ]
        actions = torch.stack([dist.sample() for dist in multi_categoricals])
        logprobs = torch.stack([dist.log_prob(a) for a, dist in zip(actions, multi_categoricals)])
        entropy = torch.stack([dist.entropy() for dist in multi_categoricals])
        num_predicted_parameters = len(envs.action_plane_space.nvec)
        actions = actions.T.view(-1, self.mapsize, num_predicted_parameters)
        logprobs = logprobs.T.view(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.mapsize, num_predicted_parameters)
        return actions, logprobs.sum(1).sum(1), entropy.sum(1).sum(1)


if __name__ == "__main__":
    args = parse_args()

    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.prod_mode:
        import wandb
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=experiment_name,
        )
        wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"runs/{experiment_name}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI],
        map_paths=args.train_maps,
        reward_weight=np.array(args.reward_weight),
        cycle_maps=args.train_maps,
    )
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = envs.reset()
    global_step = 0

    while global_step < args.total_timesteps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        invalid_action_masks = torch.tensor(envs.get_action_mask()).to(device)
        actions, logprobs, entropy = agent.get_action_and_logprob(obs_tensor, invalid_action_masks, envs)
        actions_np = actions.cpu().numpy().reshape(envs.num_envs, -1)

        next_obs, reward, done, info = envs.step(actions_np)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)

        G = reward_tensor
        returns = (G - G.mean()) / (G.std() + 1e-8)
        loss = -(logprobs * returns).mean() - args.ent_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("charts/episodic_return", reward_tensor.mean().item(), global_step)
        global_step += 1
        obs = next_obs

        if global_step % 1000 == 0:
            print(f"Step: {global_step}, Reward: {reward_tensor.mean().item():.2f}, Loss: {loss.item():.4f}")

    envs.close()
    writer.close()
