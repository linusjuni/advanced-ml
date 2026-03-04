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

TRAIN_OUTPUT=$(uv run python -m src.train_ddpm \
    --seed 42 \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-3 \
    --T 1000 \
    --device cuda 2>&1)

echo "$TRAIN_OUTPUT"

CKPT=$(echo "$TRAIN_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oP 'Run saved to \K.*' | tr -d '\r\n' | xargs)

echo "Checkpoint: $CKPT"

uv run python -m src.evaluate_pixel_ddpm \
    --ddpm-checkpoint "$CKPT" \
    --device cuda

