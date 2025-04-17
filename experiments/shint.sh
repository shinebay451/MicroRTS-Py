source ~/myenv/bin/activate


xvfb-run -a python reinforce_microrts.py \
    --total-timesteps 10000000 \
    --num-bot-envs 16 \
    --num-selfplay-envs 0 \
    --partial-obs False \
    --prod-mode \
    --capture-video
