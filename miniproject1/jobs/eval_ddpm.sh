#!/bin/bash
#BSUB -q gpuv100
#BSUB -W 2:00
#BSUB -J eval_ddpm
#BSUB -o jobs/eval_ddpm_%J.out
#BSUB -e jobs/eval_ddpm_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"

cd /zhome/e2/6/224426/project/miniproject1
mkdir -p samples

uv sync
uv run python -m src.evaluate \
    --model-path /zhome/e2/6/224426/project/miniproject1/src/checkpoints/ddpm50epoch/model.pth \
    --n-samples 1000 \
    --device cuda \
    --classifier-ckpt /zhome/e2/6/224426/project/miniproject1/src/checkpoints/mnist_classifier.pth \
    --output-dir /zhome/e2/6/224426/project/miniproject1/samples