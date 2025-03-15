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
    parser.add_argument("--agent-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument("--agent2-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="",
        help='the opponent AI to evaluate against')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    if args.ai:
        args.num_bot_envs, args.num_selfplay_envs = 1, 0
    else:
        args.num_bot_envs, args.num_selfplay_envs = 0, 2
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    return args


def decode_obs(observation):
    """
    Convert observation from one hot encoding to integer encoding
    """
    decoded_observations = []
    for unit_obs in observation: # for each grid slot
        obs = np.array([
            np.concatenate((unit_obs[0:5], np.zeros(3))),   # hit points
            np.concatenate((unit_obs[5:10], np.zeros(3))),  # resources
            np.concatenate((unit_obs[10:13], np.zeros(5))), # owner
            unit_obs[13:21],                                # unit types
            np.concatenate((unit_obs[21:27], np.zeros(2))), # current action
            np.concatenate((unit_obs[27:29], np.zeros(6))), # terrain
        ])
        decoded_obs = np.argmax(obs, axis=1)
        decoded_observations.extend(decoded_obs.tolist())

    return decoded_observations


def encode_action(action):
    """
    Convert action from integer encoding to one hot encoding
    """
    encoded_actions = []
    for unit_act in action:
        act = [
            [1 if i == unit_act[0] else 0 for i in range(6)],   # action type
            [1 if i == unit_act[1] else 0 for i in range(4)],   # move parameter
            [1 if i == unit_act[2] else 0 for i in range(4)],   # harvest parameter
            [1 if i == unit_act[3] else 0 for i in range(4)],   # return parameter
            [1 if i == unit_act[4] else 0 for i in range(4)],   # produce direction parameter
            [1 if i == unit_act[5] else 0 for i in range(7)],   # produce type parameter
            [1 if i == unit_act[6] else 0 for i in range(49)],  # relative attack position
        ]
        for a in act:
            encoded_actions.extend(a)

    return encoded_actions


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

    ais = []
    if args.ai:
        ais = [eval(f"microrts_ai.{args.ai}")]
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=args.num_selfplay_envs,
        partial_obs=False,
        max_steps=max_ep_len,
        render_theme=2,
        ai2s=ais,
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False,
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    agent = Agent(envs).to(device)
    agent2 = Agent(envs).to(device)

    # ALGO Logic: Storage for epoch data
    mapsize = 16 * 16
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())
    episodes = []

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()

    p0_name = args.agent_model_path.split("/")[-1].split(".")[0]
    save_path = f"episode_data/{p0_name}-{time.time()}".replace(".", "")
    current_save_num = 0

    # CRASH AND RESUME LOGIC:
    agent.load_state_dict(torch.load(
        args.agent_model_path, map_location=device))
    agent.eval()
    if not args.ai:
        agent2.load_state_dict(torch.load(
            args.agent2_model_path, map_location=device))
        agent2.eval()

    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)

    next_obs = torch.Tensor(envs.reset()).to(device)

    for update in tqdm(range(args.num_episodes)):
        # TRY NOT TO MODIFY: prepare the execution of the game.
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        for step in range(max_ep_len):
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                invalid_action_masks = torch.tensor(
                    np.array(envs.get_action_mask())).to(device)

                if args.ai:
                    action, _, _, _, vs = agent.get_action_and_value(
                        next_obs, envs=envs, invalid_action_masks=invalid_action_masks, device=device
                    )

                    episode_data["observations"].append(
                        decode_obs(next_obs.cpu().numpy().reshape(mapsize, -1))
                    )
                    episode_data["actions"].append(
                        encode_action(action.cpu().numpy().reshape(mapsize, -1))
                    )

                else:
                    p1_obs = next_obs[::2]
                    p2_obs = next_obs[1::2]
                    p1_mask = invalid_action_masks[::2]
                    p2_mask = invalid_action_masks[1::2]

                    episode_data["observations"].append(
                        decode_obs(p1_obs.cpu().numpy().reshape(mapsize, -1))
                    )

                    p1_action, _, _, _, _ = agent.get_action_and_value(
                        p1_obs, envs=envs, invalid_action_masks=p1_mask, device=device
                    )

                    episode_data["actions"].append(
                        encode_action(p1_action.cpu().numpy().reshape(mapsize, -1))
                    )

                    p2_action, _, _, _, _ = agent2.get_action_and_value(
                        p2_obs, envs=envs, invalid_action_masks=p2_mask, device=device
                    )
                    action = torch.zeros(
                        (args.num_envs, p2_action.shape[1], p2_action.shape[2]))
                    action[::2] = p1_action
                    action[1::2] = p2_action

            try:
                next_obs, rs, ds, infos = envs.step(
                    action.cpu().numpy().reshape(envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(device)

                episode_data["rewards"].append(float(rs[0]))
                episode_data["dones"].append(bool(ds[0]))

            except Exception as e:
                e.printStackTrace()
                raise

            # exit condition
            if ds[0]:
                break 

        episodes.append(episode_data)

        if (update + 1) % save_freq == 0:
            save_dataset(episodes, save_path, current_save_num)
            current_save_num += 1
            episodes = []

    if len(episodes) > 0:
        save_dataset(episodes, current_save_num)

    envs.close()
