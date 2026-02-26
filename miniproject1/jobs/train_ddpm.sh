#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 8:00
#BSUB -J ddpm_unet
#BSUB -o jobs/ddpm_unet_%J.out
#BSUB -e jobs/ddpm_unet_%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

uv sync
uv run python -m src.train_ddpm \
    --seed 42 \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3 \
    --T 1000 \
    --device cuda

