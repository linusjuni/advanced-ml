#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 4:00
#BSUB -J vae_flow
#BSUB -o jobs/vae_flow_%J.out
#BSUB -e jobs/vae_flow_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

uv sync
uv run python -m src.train_vae flow \
    --seed 1 \
    --M 32 \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3 \
    --device cuda \
    --flow-num-layers 5 \
    --flow-num-hidden 64
