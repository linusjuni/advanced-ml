#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -J mp2_partb_geodesics
#BSUB -o miniproject2/jobs/mp2_partb_geodesics_%J.out
#BSUB -e miniproject2/jobs/mp2_partb_geodesics_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail

# ---------- Config ----------
DEVICE="cuda"
BATCH_SIZE=32
LATENT_DIM=2

NUM_CURVES=25
NUM_T=10 # specified by exercise
MC_SAMPLES=10 # no magic reason for this, just a reasonable number to get decent estimates without taking forever

# Same sweep as training runs
DECODER_COUNTS=(1 2 3)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

BASE_DIR="miniproject2/experiments/part_b"
# ----------------------------

uv sync

for NUM_DECODERS in "${DECODER_COUNTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    EXP_DIR="${BASE_DIR}/m${NUM_DECODERS}_seed${SEED}"

    if [[ ! -f "${EXP_DIR}/model.pt" ]]; then
      echo "Skipping ${EXP_DIR}: model.pt not found"
      continue
    fi

    echo "============================================================"
    echo "Computing geodesics: num_decoders=${NUM_DECODERS}, seed=${SEED}"
    echo "Using checkpoint: ${EXP_DIR}/model.pt"

    uv run miniproject2/part_b.py geodesics \
      --num-decoders "${NUM_DECODERS}" \
      --seed "${SEED}" \
      --batch-size "${BATCH_SIZE}" \
      --latent-dim "${LATENT_DIM}" \
      --num-curves "${NUM_CURVES}" \
      --num-t "${NUM_T}" \
      --mc-samples "${MC_SAMPLES}" \
      --device "${DEVICE}" \
      --experiment-folder "${EXP_DIR}"
  done
done

echo "All Part B geodesic runs completed successfully."
