#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 4:00
#BSUB -J latent_ddpm
#BSUB -o jobs/latent_ddpm_%J.out
#BSUB -e jobs/latent_ddpm_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

uv sync

# Train VAE and capture the output to find checkpoint path
echo "Training VAE..."
VAE_OUTPUT=$(uv run python -m src.train_vae gaussian \
    --seed 1 \
    --M 32 \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-3 \
    --decoder-type gaussian \
    --beta 1e-6 \
    --device cuda 2>&1)

echo "$VAE_OUTPUT"

VAE_CHECKPOINT=$(echo "$VAE_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oP 'Run saved to \K.*' | tr -d '\r\n' | xargs)

if [ -z "$VAE_CHECKPOINT" ]; then
    echo "Error: Could not find VAE checkpoint path in output"
    exit 1
fi

echo "VAE checkpoint saved to: $VAE_CHECKPOINT"

# Train latent DDPM using the VAE checkpoint
echo "Training latent DDPM..."
LATENT_OUTPUT=$(uv run python -m src.train_latent_ddpm \
    --seed 1 \
    --M 32 \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --device cuda \
    --T 1000 \
    --pretrained-vae-checkpoint "$VAE_CHECKPOINT")
echo "Training complete!"

LATENT_CHECKPOINT=$(echo "$LATENT_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oP 'Run saved to \K.*' | tr -d '\r\n' | xargs)

# Evaluate and visualize results
echo "Evaluating and visualizing results..."
uv run python -m src.evaluate_ddpm --vae-checkpoint "$VAE_CHECKPOINT" --ddpm-checkpoint "$LATENT_CHECKPOINT" 
echo "Evaluation and visualization complete!"