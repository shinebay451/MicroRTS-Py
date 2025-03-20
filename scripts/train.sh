#!/bin/sh
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00

cd "$HOME/Documents/repositories/MicroRTS-Py" || exit
source .venv/bin/activate
cd ./experiments

python ppo_gridnet.py \
  --total-timesteps 200000000 \
  --seed 1 \
  --prod-mode \
  --wandb-project-name microrts-py \
  --exp-name rewards-parametrization-27 \
  --reward-weight 8.736320496 2.818786621 0.794951081 1.050905228 1.104272008 4.690114021
