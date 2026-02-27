import csv
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from utils.model_utils import load_model
from dataset import get_binarized_mnist, get_dequantized_mnist


def visualize_training_curves(vae_checkpoint_dir: str | Path, ddpm_checkpoint_dir: str | Path):
    """Plot training loss curves from metrics.csv for both VAE and DDPM"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot VAE training curve
    vae_metrics_path = Path(vae_checkpoint_dir) / "metrics.csv"
    if vae_metrics_path.exists():
        epochs, losses = [], []
        with open(vae_metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                losses.append(float(row['train_loss']))
        
        ax1.plot(epochs, losses, marker='o', linewidth=2, markersize=4, color='blue')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('VAE Training Loss', fontsize=14)
        ax1.grid(True, alpha=0.3)
    
    # Plot DDPM training curve
    ddpm_metrics_path = Path(ddpm_checkpoint_dir) / "metrics.csv"
    if ddpm_metrics_path.exists():
        epochs, losses = [], []
        with open(ddpm_metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                losses.append(float(row['train_loss']))
        
        ax2.plot(epochs, losses, marker='o', linewidth=2, markersize=4, color='red')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('Latent DDPM Training Loss', fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(ddpm_checkpoint_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved training curves to {output_path}")
    plt.close()


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
    
    # Reshape samples to 28x28
    samples = samples.reshape(-1, 28, 28)
    
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
    # Load models
    parser = argparse.ArgumentParser(description="Predict Latent DDPM on MNIST")

    parser.add_argument("--vae-checkpoint", type=str, help="Path to VAE checkpoint directory")
    parser.add_argument("--ddpm-checkpoint", type=str, help="Path to DDPM checkpoint directory")

    args = parser.parse_args()

    vae_checkpoint_dir = args.vae_checkpoint
    diffusion_checkpoint_dir = args.ddpm_checkpoint
    print("Loading VAE...")
    vae, vae_config = load_model(vae_checkpoint_dir)
    print(f"Loaded VAE with config: {vae_config}")
    
    print("\nLoading Latent DDPM...")
    diffusion_model, ddpm_config = load_model(diffusion_checkpoint_dir)
    print(f"Loaded DDPM with config: {ddpm_config}")
    print(f"Latent dimension: {ddpm_config.M}")
    
    # Load test data
    test_dataset = get_dequantized_mnist(train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    print("\nGenerating visualizations...")
    
    # Generate all visualizations
    visualize_training_curves(vae_checkpoint_dir, diffusion_checkpoint_dir)
    visualize_samples(diffusion_model, n_samples=64, checkpoint_dir=diffusion_checkpoint_dir, vae=vae)
    
    print("\nAll visualizations complete!")