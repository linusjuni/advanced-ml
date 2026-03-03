#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 2:00
#BSUB -J eval_vae
#BSUB -o jobs/eval_vae_%J.out
#BSUB -e jobs/eval_vae_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

cd /zhome/e2/6/224426/project/advanced-ml/miniproject1
mkdir -p samples

uv sync
uv run python -m src.evaluate_vae_mat \
    --model-path /zhome/e2/6/224426/project/advanced-ml/miniproject1/src/checkpoints/vae_gaussian_mat/model.pth \
    --n-samples 1000 \
    --device cuda \
    --classifier-ckpt /zhome/e2/6/224426/project/advanced-ml/miniproject1/src/checkpoints/mnist_classifier.pth \
    --output-dir /zhome/e2/6/224426/project/advanced-ml/miniproject1/samples