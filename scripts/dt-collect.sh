#!/bin/sh
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00

cd "$HOME/Documents/repositories/MicroRTS-Py" || exit
source .venv/bin/activate
cd ./experiments

python dt_gridnet_collect.py --agent-model-path models/test-agent.pt --ai coacAI --num-episodes 1000
