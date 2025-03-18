# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from classes.DataCollector import DecisionTransformerGymDataCollator
from classes.TrainableDT import TrainableDT
from datasets import DatasetDict
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai  # noqa


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument("--agent-model-path", type=str, help="the path to the agent's model")
    parser.add_argument("--agent2-model-path", type=str, help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="",
        help='the opponent AI to evaluate against')
    parser.add_argument('--num-episodes', type=int, default=10, help='number of episodes to play')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    if args.ai:
        args.num_bot_envs, args.num_selfplay_envs = 1, 0
    else:
        args.num_bot_envs, args.num_selfplay_envs = 0, 2
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


def get_action(model, states, actions, returns_to_go, timesteps, invalid_action_mask):
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length:]
    actions = actions[:, -model.config.max_length:]
    returns_to_go = returns_to_go[:, -model.config.max_length:]
    timesteps = timesteps[:, -model.config.max_length:]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat(
        [torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat(
        [torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat(
        [torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat(
        [torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat(
        [torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    action_pred = model.original_forward(
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        invalid_action_mask=invalid_action_mask,
        return_dict=False,
    )

    return action_pred


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


def decode_action(action: torch.Tensor):
    """
    Convert action from integer encoding to one hot encoding.
    """
    decoded_action = torch.zeros(
        (action.size(0), 7),
        dtype=torch.int,
        device=action.device
    )
    slices = [
        (0, 6),    # action type
        (6, 10),   # move parameter
        (10, 14),  # harvest parameter
        (14, 18),  # return parameter
        (18, 22),  # produce direction parameter
        (22, 29),  # produce type parameter
        (29, 78)   # relative attack position
    ]
    paddings = [
        43,  # action type
        45,  # move parameter
        45,  # harvest parameter
        45,  # return parameter
        45,  # produce direction parameter
        42,  # produce type parameter
        0    # relative attack position
    ]
    current_pos = 0

    for unit_action in action:  # for each grid slot
        act = []
        for (start, end), pad in zip(slices, paddings):
            component = unit_action[start:end]
            if pad > 0:
                component = torch.cat(
                    (component, torch.zeros(pad, device=action.device))
                )
            act.append(component)

        act = torch.stack(act)
        decoded_act = torch.argmax(act, dim=1)
        decoded_action[current_pos, :] = decoded_act
        current_pos += 1

    return decoded_action.flatten()


if __name__ == "__main__":
    args = parse_args()

    from ppo_gridnet import Agent, MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.prod_mode:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        CHECKPOINT_FREQUENCY = 10
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters", "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # TRY NOT TO MODIFY: seeding
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # variables for decision transformer
    TARGET_RETURN = 10
    dataset = DatasetDict.load_from_disk(
        "episode_data/cm-mcrp-1742170770231193/save_0")
    collector = DecisionTransformerGymDataCollator(dataset["train"])

    ais = []
    if args.ai:
        ais = [eval(f"microrts_ai.{args.ai}")]
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=args.num_selfplay_envs,
        partial_obs=args.partial_obs,
        max_steps=collector.max_ep_len,
        render_theme=2,
        ai2s=ais,
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs, f"videos/{experiment_name}", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000
        )
    assert isinstance(
        envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    agent = TrainableDT.from_pretrained(args.agent_model_path).to(device)
    agent2 = Agent(envs).to(device)

    # ALGO Logic: Storage for epoch data
    mapsize = 16 * 16
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)

    agent.eval()
    if not args.ai:
        agent2.load_state_dict(torch.load(
            args.agent2_model_path, map_location=device))
        agent2.eval()

    for update in range(args.num_episodes):
        # TRY NOT TO MODIFY: prepare the execution of the game.
        states = decode_obs(next_obs[0].view(mapsize, -1)).reshape(
            1, collector.state_dim).to(device=device, dtype=torch.float32)
        target_return = torch.tensor(
            TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
        actions = torch.zeros(
            (0, collector.act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        timesteps = torch.tensor(
            0, device=device, dtype=torch.long).reshape(1, 1)

        for step in range(collector.max_ep_len):
            envs.render(mode="rgb_array")
            with torch.no_grad():
                invalid_action_masks = torch.tensor(
                    np.array(envs.get_action_mask())).to(device)

                actions = torch.cat([actions, torch.zeros(
                    (1, collector.act_dim), device=device)], dim=0)
                rewards = torch.cat(
                    [rewards, torch.zeros(1, device=device)])

                p1_action = get_action(
                    agent,
                    (states - collector.state_mean) / collector.state_std,
                    actions,
                    target_return,
                    timesteps,
                    invalid_action_masks[0]
                )
                actions[-1] = p1_action

                action = torch.zeros(
                    (args.num_envs, mapsize, 7), device=device)

                action[::2] = decode_action(
                    p1_action.view(mapsize, -1)).view(1, mapsize, -1)

                if args.agent2_model_path:
                    p2_obs = next_obs[1::2]
                    p2_mask = invalid_action_masks[1::2]

                    p2_action, _, _, _, _ = agent2.get_action_and_value(
                        p2_obs, envs=envs, invalid_action_masks=p2_mask, device=device
                    )
                    action[1::2] = p2_action

                action = action.detach().cpu().numpy()

                try:
                    next_obs, rs, ds, infos = envs.step(
                        action.reshape(envs.num_envs, -1))
                    next_obs = torch.Tensor(next_obs).to(device)
                except Exception as e:
                    e.printStackTrace()
                    raise

                next_state = torch.Tensor(
                    decode_obs(next_obs[0].view(mapsize, -1))
                ).to(device).reshape(1, collector.state_dim)

                states = torch.cat([states, next_state], dim=0)
                rewards[-1] = rs[0]

                pred_return = target_return[0, -1] - rs[0]
                target_return = torch.cat(
                    [target_return, pred_return.reshape(1, 1)], dim=1)
                timesteps = torch.cat([timesteps, torch.ones(
                    (1, 1), device=device, dtype=torch.long) * (step + 1)], dim=1)

                for idx, info in enumerate(infos):
                    if "episode" in info.keys():
                        if args.ai:
                            print("against", args.ai,
                                  info["microrts_stats"]["WinLossRewardFunction"])
                        else:
                            if idx % 2 == 0:
                                print(
                                    f"player{idx % 2}", info["microrts_stats"]["WinLossRewardFunction"])

                if ds[0]:
                    break

    envs.close()
    writer.close()
