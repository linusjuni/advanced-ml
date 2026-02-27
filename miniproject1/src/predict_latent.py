import csv
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from utils.model_utils import load_model
from dataset import get_binarized_mnist


def visualize_samples(model, n_samples=64, checkpoint_dir=None, vae=None):
    """Generate and visualize samples from the model"""
    model.eval()
    if vae:
        vae.eval()
    
    with torch.no_grad():
        # Sample latent vectors from DDPM
        latent_samples = model.sample(n_samples)
        
        if vae:
            # Decode latent vectors to images using VAE decoder
            decoded_dist = vae.decoder(latent_samples)
            samples = decoded_dist.sample().cpu().numpy()
        else:
            samples = latent_samples.cpu().numpy()
    
    # Create grid of samples
    n_rows = int(np.sqrt(n_samples))
    n_cols = n_samples // n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(samples[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    
    plt.suptitle(f'Generated Samples (n={n_samples})', fontsize=16)
    plt.tight_layout()
    
    if checkpoint_dir:
        plt.savefig(Path(checkpoint_dir) / 'generated_samples.png', dpi=150)
        print(f"Saved generated samples to {checkpoint_dir}/generated_samples.png")
    plt.close()


if __name__ == "__main__":
    # Load model
    checkpoint_dir = "src/checkpoints/vae_M32_priorgaussian_seed1_lr0.001_bs128_ep20_20260227_104346"
    vae, config = load_model(checkpoint_dir)

    diffusion_model_dir = "src/checkpoints/latent_ddpm_M32_num_hidden512_T1000_beta_10.0001_beta_T0.02_seed1_lr0.001_bs64_ep100_20260227_124518/model.pth"
    diffusion_model, config = load_model(diffusion_model_dir)
    
    
    print(f"Loaded model with config: {config}")
    print(f"Latent dimension: {config.M}")
    
    # Load test data
    test_dataset = get_binarized_mnist(train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    print("\nGenerating visualizations...")
    
    # Generate all visualizations
    visualize_samples(diffusion_model, n_samples=64, checkpoint_dir=checkpoint_dir, vae=vae)
    
    print("\nAll visualizations complete!")