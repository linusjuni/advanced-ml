#!/bin/bash
#BSUB -q gpuv100
#BSUB -J eval_gvae
#BSUB -W 01:00
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -o jobs/logs/eval_gvae_%J.out
#BSUB -e jobs/logs/eval_gvae_%J.err

# Usage: CHECKPOINT=outputs/train_gvae/<timestamp>/best_model.pt bsub < jobs/run_evaluation.sh
: "${CHECKPOINT:?Set CHECKPOINT env var before submitting}"

uv run -m entrypoints.run_evaluation --checkpoint "$CHECKPOINT" --device cpu