#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -J mp2_partb_ensemble
#BSUB -o miniproject2/jobs/mp2_partb_ensemble_%J.out
#BSUB -e miniproject2/jobs/mp2_partb_ensemble_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail

# ---------- Config ----------
DEVICE="cuda"
EPOCHS=50
BATCH_SIZE=32
LATENT_DIM=2

# Part B requirement: use 1, 2, 3 decoders and at least 10 retrainings.
DECODER_COUNTS=(1 2 3)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

BASE_DIR="miniproject2/experiments/part_b"
# ----------------------------

uv sync

for NUM_DECODERS in "${DECODER_COUNTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    EXP_DIR="${BASE_DIR}/m${NUM_DECODERS}_seed${SEED}"

    echo "============================================================"
    echo "Training ensemble VAE: num_decoders=${NUM_DECODERS}, seed=${SEED}"
    echo "Saving to: ${EXP_DIR}"

    uv run miniproject2/part_b.py train \
      --num-decoders "${NUM_DECODERS}" \
      --seed "${SEED}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --latent-dim "${LATENT_DIM}" \
      --device "${DEVICE}" \
      --experiment-folder "${EXP_DIR}"
  done
done

echo "All Part B training runs completed successfully."
