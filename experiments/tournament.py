# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from gym_microrts import microrts_ai  # noqa


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument(
        '--games-per-match', 
        type=int,
        default=10,
        help='the number of games to be played in a match'
    )
    parser.add_argument(
        "--agents",
        nargs='+',
        help="the path to the agent models to be ran in the tournament"
    )
    parser.add_argument(
        '--ais', 
        nargs='+',
        help="the ais to be ran in the tournament"
    )

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    return args


def get_bot_name(bot, bot_type):
    return bot["path"].split("/")[-1].split(".")[0] \
            if bot_type == "agent" else bot["classname"]


def print_final_results(bots_wins, all_bots):
    print("\nFinal Results:")
    bot_names = [get_bot_name(bot, bot["type"]) for bot in all_bots]
    header = " " * 15 + " | " + " | ".join(f"{name:>15}" for name in bot_names) + " | Total Wins"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(bots_wins):
        total_wins = sum(row)
        row_str = f"{bot_names[i]:>15} | " + " | ".join(f"{win:>15}" for win in row) + f" | {total_wins:>10}"
        print(row_str)


if __name__ == "__main__":
    args = parse_args()

    from ppo_gridnet import Agent, MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import (MicroRTSBotVecEnv,
                                           MicroRTSGridModeVecEnv)

    # TRY NOT TO MODIFY: seeding
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    max_ep_length = 2000
    exp_name = f"tournament-{time.time()}".replace(".", "")

    all_bots = []
    if args.agents:
        all_bots.extend(
            [{"type": "agent", "path": agent} for agent in args.agents]
        )
    if args.ais:
        all_bots.extend(
            [{"type": "ai", "classname": ai} for ai in args.ais]
        )
    assert len(all_bots) > 1, "at least 2 agents/ais are required to play a tournament"

    bots_wins = np.zeros((len(all_bots), len(all_bots)), dtype=np.int32)

    for i in range(len(all_bots)):
        for j in range(i + 1, len(all_bots)):
            player1_type = all_bots[i]["type"]
            player2_type = all_bots[j]["type"]
            ai1s = None
            ai2s = None

            p1_name = all_bots[i]["path"].split("/")[-1].split(".")[0] \
                    if player1_type == "agent" else all_bots[i]["classname"]
            p2_name = all_bots[j]["path"].split("/")[-1].split(".")[0] \
                    if player2_type == "agent" else all_bots[j]["classname"]

            if player1_type == "agent" and player2_type == "agent":
                bot_envs, selfplay_envs = 0, 2 
                mode = 0
            elif player1_type == "ai" and player2_type == "ai":
                bot_envs, selfplay_envs = 1, 0
                mode = 2
                ai1s = [eval(f"microrts_ai.{all_bots[i]['classname']}")]
                ai2s = [eval(f"microrts_ai.{all_bots[j]['classname']}")]
            else:
                bot_envs, selfplay_envs = 1, 0
                mode = 1

                if player1_type == "ai":
                    ai2s = [eval(f"microrts_ai.{all_bots[i]['classname']}")]
                else:
                    ai2s = [eval(f"microrts_ai.{all_bots[j]['classname']}")]

            if mode == 0:
                envs = MicroRTSGridModeVecEnv(
                    num_bot_envs=bot_envs,
                    num_selfplay_envs=selfplay_envs,
                    partial_obs=False,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=["maps/16x16/basesWorkers16x16A.xml"],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )

            elif mode == 1:
                envs = MicroRTSGridModeVecEnv(
                    num_bot_envs=bot_envs,
                    num_selfplay_envs=selfplay_envs,
                    ai2s=ai2s,
                    partial_obs=False,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=["maps/16x16/basesWorkers16x16A.xml"],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )

            else:
                envs = MicroRTSBotVecEnv(
                    ai1s=ai1s,
                    ai2s=ai2s,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=["maps/16x16/basesWorkers16x16A.xml"],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )

            envs = MicroRTSStatsRecorder(envs)
            envs = VecMonitor(envs)

            if args.capture_video:
                envs = VecVideoRecorder(
                    envs, 
                    f"videos/{exp_name}/{p1_name}-{p2_name}",
                    record_video_trigger=lambda x: x == 0,
                    video_length=max_ep_length
                )

            if mode == 0:
                agent1 = Agent(envs).to(device)
                agent2 = Agent(envs).to(device)

                agent1.load_state_dict(torch.load(all_bots[i]["path"], map_location=device), strict=False)
                agent1.eval()
                agent2.load_state_dict(torch.load(all_bots[j]["path"], map_location=device), strict=False)
                agent2.eval()

                next_obs = torch.Tensor(envs.reset()).to(device)
            elif mode == 1:
                agent1 = Agent(envs).to(device)

                if player1_type == "agent":
                    agent1.load_state_dict(torch.load(all_bots[i]["path"], map_location=device), strict=False)
                else:
                    agent1.load_state_dict(torch.load(all_bots[j]["path"], map_location=device), strict=False)
                agent1.eval()

                next_obs = torch.Tensor(envs.reset()).to(device)
            else:
                next_obs = envs.reset()

            print("\n\n====== Next Match ======")
            print(f"{p1_name} vs. {p2_name}")

            for game in range(args.games_per_match):
                for update in range(max_ep_length):
                    envs.render(mode="rgb_array")
                    if mode in [0, 1]:
                        with torch.no_grad():
                            invalid_action_masks = torch.tensor(
                                np.array(envs.get_action_mask())).to(device)

                            if mode == 1:
                                action, _, _, _, _ = agent1.get_action_and_value(
                                    next_obs, envs=envs, invalid_action_masks=invalid_action_masks, device=device
                                )

                            else:

                                p1_obs = next_obs[::2]
                                p2_obs = next_obs[1::2]
                                p1_mask = invalid_action_masks[::2]
                                p2_mask = invalid_action_masks[1::2]

                                p1_action, _, _, _, _ = agent1.get_action_and_value(
                                    p1_obs, envs=envs, invalid_action_masks=p1_mask, device=device
                                )
                                p2_action, _, _, _, _ = agent2.get_action_and_value(
                                    p2_obs, envs=envs, invalid_action_masks=p2_mask, device=device
                                )
                                action = torch.zeros(
                                    (bot_envs + selfplay_envs, p2_action.shape[1], p2_action.shape[2]))
                                action[::2] = p1_action
                                action[1::2] = p2_action

                        try:
                            next_obs, _, ds, infos = envs.step(
                                action.cpu().numpy().reshape(envs.num_envs, -1))
                            next_obs = torch.Tensor(next_obs).to(device)
                        except Exception as e:
                            e.printStackTrace()
                            raise

                    else:
                        # dummy action
                        next_obs, _, ds, infos = envs.step(
                            [
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ]
                        )

                    for idx, info in enumerate(infos):
                        if "episode" in info.keys():
                            score = int(info["microrts_stats"]["WinLossRewardFunction"])

                            if mode == 0:
                                if idx % 2 == 0:
                                    if score > 0:
                                        bots_wins[i][j] += 1
                                        print(f"Game {game + 1}: {p1_name} wins!")

                                    elif score == 0:
                                        print(f"Game {game + 1}: it's a draw!")

                                elif score > 0:
                                    bots_wins[j][i] += 1
                                    print(f"Game {game + 1}: {p2_name} wins!")

                            elif mode == 1:
                                if player1_type == "agent":
                                    if score > 0:
                                        bots_wins[i][j] += 1
                                        print(f"Game {game + 1}: {p1_name} wins!")

                                    elif score < 0:
                                        bots_wins[j][i] += 1
                                        print(f"Game {game + 1}: {p2_name} wins!")

                                    else:
                                        print(f"Game {game + 1}: it's a draw!")

                                else:
                                    if score > 0:
                                        bots_wins[j][i] += 1
                                        print(f"Game {game + 1}: {p2_name} wins!")

                                    elif score < 0:
                                        bots_wins[i][j] += 1
                                        print(f"Game {game + 1}: {p1_name} wins!")

                                    else:
                                        print(f"Game {game + 1}: it's a draw!")

                            else:
                                if score > 0:
                                    bots_wins[i][j] += 1
                                    print(f"Game {game + 1}: {p1_name} wins!")

                                elif score < 0:
                                    bots_wins[j][i] += 1
                                    print(f"Game {game + 1}: {p2_name} wins!")

                                else:
                                    print(f"Game {game + 1}: it's a draw!")

                    # game exit condition
                    if ds[0]:
                        break;

            print("====== Match Done ======\n\n")

    print_final_results(bots_wins, all_bots)
