#!/bin/sh
#SBATCH --gpus-per-node=1
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

cd ~/MicroRTS-Py

source .venv/bin/activate

export JAVA_HOME=/usr/lib/jvm/jre/

uv pip install -r requirements.txt

pwd > -venv/lib/python3.9/site-packages/gym_microrts.pth
python hello_world.py

cd experiments
python REINFORCE.py
