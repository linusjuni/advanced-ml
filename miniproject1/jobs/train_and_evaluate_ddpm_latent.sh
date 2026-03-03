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

# Beta values to sweep
BETAS=(1e-6 1e-3 1 10)

for BETA in "${BETAS[@]}"; do
    echo "Running pipeline with beta=$BETA"

    # Train VAE and capture the output to find checkpoint path
    echo "Training VAE (beta=$BETA)..."
    VAE_OUTPUT=$(uv run python -m src.train_vae gaussian \
        --seed 1 \
        --M 32 \
        --epochs 50 \
        --batch-size 128 \
        --lr 1e-3 \
        --decoder-type gaussian \
        --beta "$BETA" \
        --device cuda 2>&1)

    echo "$VAE_OUTPUT"

    VAE_CHECKPOINT=$(echo "$VAE_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oP 'Run saved to \K.*' | tr -d '\r\n' | xargs)

    if [ -z "$VAE_CHECKPOINT" ]; then
        echo "Error: Could not find VAE checkpoint path in output for beta=$BETA"
        exit 1
    fi

    echo "VAE checkpoint saved to: $VAE_CHECKPOINT"

    # Train latent DDPM using the VAE checkpoint
    echo "Training latent DDPM (beta=$BETA)..."
    LATENT_OUTPUT=$(uv run python -m src.train_latent_ddpm \
        --beta "$BETA" \
        --seed 1 \
        --M 32 \
        --epochs 100 \
        --batch-size 64 \
        --lr 1e-3 \
        --device cuda \
        --T 1000 \
        --pretrained-vae-checkpoint "$VAE_CHECKPOINT" 2>&1)
    echo "$LATENT_OUTPUT"
    echo "Training complete for beta=$BETA!"

    LATENT_CHECKPOINT=$(echo "$LATENT_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oP 'Run saved to \K.*' | tr -d '\r\n' | xargs)

    if [ -z "$LATENT_CHECKPOINT" ]; then
        echo "Error: Could not find latent DDPM checkpoint path in output for beta=$BETA"
        exit 1
    fi

    # Evaluate and visualize results
    echo "Evaluating and visualizing results for beta=$BETA..."
    uv run python -m src.evaluate_ddpm --vae-checkpoint "$VAE_CHECKPOINT" --ddpm-checkpoint "$LATENT_CHECKPOINT" --beta "$BETA"
    echo "Evaluation and visualization complete for beta=$BETA!"
done

echo "All runs completed successfully."