import argparse

import torch

from src.vae import VAE
from src.ddpm import DDPM
from src.utils.model_utils import load_model
from src.utils.viz_utils import plot_training_curves, plot_sample_grid
from src.utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Latent DDPM on MNIST")
    parser.add_argument("--vae-checkpoint", type=str, help="Path to VAE checkpoint directory")
    parser.add_argument("--ddpm-checkpoint", type=str, help="Path to DDPM checkpoint directory")
    args = parser.parse_args()

    vae_checkpoint_dir = args.vae_checkpoint
    diffusion_checkpoint_dir = args.ddpm_checkpoint

    logger.info("Loading VAE...")
    vae, vae_config = load_model(vae_checkpoint_dir)
    assert isinstance(vae, VAE)
    logger.info(f"Loaded VAE with config: {vae_config}")

    logger.info("Loading Latent DDPM...")
    diffusion_model, ddpm_config = load_model(diffusion_checkpoint_dir)
    assert isinstance(diffusion_model, DDPM)
    logger.info(f"Loaded DDPM with config: {ddpm_config}")

    # Training curves
    plot_training_curves(
        [f"{vae_checkpoint_dir}/metrics.csv", f"{diffusion_checkpoint_dir}/metrics.csv"],
        labels=["VAE", "Latent DDPM"],
        save_path=f"{diffusion_checkpoint_dir}/training_curves.png",
    )

    # Generate samples
    vae.eval()
    diffusion_model.eval()
    with torch.no_grad():
        latent_samples = diffusion_model.sample(64)
        samples = vae.decoder(latent_samples).sample().cpu().numpy()

    plot_sample_grid(
        samples,
        title="Latent DDPM Samples",
        save_path=f"{diffusion_checkpoint_dir}/generated_samples.png",
    )

    logger.success("All visualizations complete!")
