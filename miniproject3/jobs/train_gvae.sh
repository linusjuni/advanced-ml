#!/bin/bash
#BSUB -q gpuv100
#BSUB -J train_gvae
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o jobs/logs/train_gvae_%J.out
#BSUB -e jobs/logs/train_gvae_%J.err

uv run -m entrypoints.train_gvae --epochs 1000 --batch-size 32 --kl-warmup-epochs 50