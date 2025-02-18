#!/bin/sh

cd "/home/cm787623/Documents/repositories/MicroRTS-Py/experiments" || exit
source /home/cm787623/.cache/pypoetry/virtualenvs/gym-microrts-ZnOR3e21-py3.9/bin/activate

python ppo_gridnet.py \
  --total-timesteps 100000000 \
  --seed 1 \
  --prod-mode \
  --wandb-project-name microrts-py \
  --exp-name rewards-parametrization-27 \
  --train-maps maps/16x16/basesWorkers16x16B.xml maps/16x16/basesWorkers16x16C.xml maps/16x16/basesWorkers16x16D.xml maps/16x16/basesWorkers16x16E.xml maps/16x16/basesWorkers16x16F.xml \
  --reward-weight 8.736320496 2.818786621 0.794951081 1.050905228 1.104272008 4.690114021

deactivate
