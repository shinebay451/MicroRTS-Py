import argparse
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import tqdm

from gym_microrts import microrts_ai  # noqa


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--num-episodes', type=int, default=100, help='number of episodes to save in dataset')

    # Algorithm specific arguments
    parser.add_argument("--agent-model-path", type=str, default='gym-microrts-static-files/agent_sota.pt',
                        help="the path to the agent's model")
    parser.add_argument('--ai2s', nargs='+', type=str, 
                        default=['coacAI', 'agentP', 'workerRushAI', 'randomBiasedAI', 'lightRushAI', 'naiveMCTSAI', 'mayari'],
                        help='the opponent AIs to evaluate against')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

    return args


def decode_obs(observation: torch.Tensor):
    """
    Convert observation from one hot encoding to integer encoding.
    """
    decoded_observations = torch.zeros(
        (observation.size(0), 6),
        dtype=torch.int,
        device=observation.device
    )
    slices = [
        (0, 5),    # hit points
        (5, 10),   # resources
        (10, 13),  # owner
        (13, 21),  # unit types
        (21, 27),  # current action
        (27, 29)   # terrain
    ]
    paddings = [
        3,  # hit points
        3,  # resources
        5,  # owner
        0,  # unit types
        2,  # current action
        6   # terrain
    ]
    current_pos = 0

    for unit_obs in observation:  # for each grid slot
        obs = []
        for (start, end), pad in zip(slices, paddings):
            component = unit_obs[start:end]
            if pad > 0:
                component = torch.cat(
                    (component, torch.zeros(pad, device=observation.device))
                )
            obs.append(component)

        obs = torch.stack(obs)
        decoded_obs = torch.argmax(obs, dim=1)
        decoded_observations[current_pos, :] = decoded_obs
        current_pos += 1

    return decoded_observations.flatten()


def encode_action(action: torch.Tensor):
    """
    Convert action from integer encoding to one hot encoding.
    """
    sizes = [6, 4, 4, 4, 4, 7, 49]
    encoded_actions = torch.zeros(
        (action.size(0), sum(sizes)),
        dtype=torch.int,
        device=action.device
    )
    current_pos = 0

    for i, size in enumerate(sizes):
        one_hot = torch.nn.functional.one_hot(action[:, i], num_classes=size).float()
        encoded_actions[:, current_pos:current_pos + size] = one_hot
        current_pos += size

    return encoded_actions.flatten()


def save_dataset(episodes, save_path, save_num):
    """
    Save the episodes to a dataset
    """
    data_dict = {
        "observations": [episode["observations"] for episode in episodes],
        "actions": [episode["actions"] for episode in episodes],
        "rewards": [episode["rewards"] for episode in episodes],
        "dones": [episode["dones"] for episode in episodes],
    }
    new_dataset = Dataset.from_dict(data_dict)

    prev_save_path = f"{save_path}/save_{save_num-1}"
    next_save_path = f"{save_path}/save_{save_num}"

    try:
        existing_dataset = DatasetDict.load_from_disk(prev_save_path)
        combined_dataset = DatasetDict({
            "train": concatenate_datasets([existing_dataset["train"], new_dataset])
        })
    except:
        combined_dataset = DatasetDict({"train": new_dataset})

    combined_dataset.save_to_disk(next_save_path)


if __name__ == "__main__":
    args = parse_args()

    from ppo_gridnet import Agent, MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

    # TRY NOT TO MODIFY: seeding
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    max_ep_len = 2000
    save_freq = 100
    episodes = []

    p0_name = args.agent_model_path.split("/")[-1].split(".")[0]
    save_path = f"episode_data/{p0_name}-{time.time()}".replace(".", "")
    current_save_num = 0
    mapsize = 16 * 16

    for update in tqdm(range(args.num_episodes)):
        ai2 = random.choice(args.ai2s)
        ai2 = [eval(f"microrts_ai.{ai2}")]

        envs = MicroRTSGridModeVecEnv(
            num_bot_envs=1,
            num_selfplay_envs=0,
            partial_obs=False,
            max_steps=max_ep_len,
            render_theme=2,
            ai2s=ai2,
            map_paths=["maps/16x16/basesWorkers16x16A.xml"],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            autobuild=False,
        )

        agent = Agent(envs).to(device)
        agent.load_state_dict(torch.load(
            args.agent_model_path, map_location=device))
        agent.eval()

        envs = MicroRTSStatsRecorder(envs)
        envs = VecMonitor(envs)
        next_obs = torch.Tensor(envs.reset()).to(device)

        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        for step in range(max_ep_len):
            with torch.no_grad():
                invalid_action_masks = torch.tensor(
                    np.array(envs.get_action_mask())).to(device)

                episode_data["observations"].append(
                    decode_obs(next_obs.view(mapsize, -1))
                )

                action, _, _, _, vs = agent.get_action_and_value(
                    next_obs, envs=envs, invalid_action_masks=invalid_action_masks, device=device
                )

                episode_data["actions"].append(
                    encode_action(action.view(mapsize, -1))
                )

                try:
                    next_obs, rs, ds, infos = envs.step(
                        action.cpu().numpy().reshape(envs.num_envs, -1))
                    next_obs = torch.Tensor(next_obs).to(device)

                except Exception as e:
                    e.printStackTrace()
                    raise

                episode_data["rewards"].append(float(rs[0]))
                episode_data["dones"].append(bool(ds[0]))


            # exit condition
            if ds[0]:
                break 

        episodes.append(episode_data)

        if (update + 1) % save_freq == 0:
            save_dataset(episodes, save_path, current_save_num)
            current_save_num += 1
            episodes = []

    if len(episodes) > 0:
        save_dataset(episodes, save_path, current_save_num)

    envs.close()
