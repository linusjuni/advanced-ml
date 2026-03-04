import argparse
import time

import matplotlib.pyplot as plt
import torch

from src.vae import VAE
from src.ddpm import DDPM
from src.dataset import get_dequantized_mnist
from src.fid import compute_fid
from src.utils.model_utils import load_model, CHECKPOINTS_DIR
from src.utils.viz_utils import plot_training_curves
from src.utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Latent DDPM on MNIST")
    parser.add_argument("--vae-checkpoint", type=str, help="Path to VAE checkpoint directory")
    parser.add_argument("--beta", type=float, required=True, help="Beta value used for training the VAE (for labeling purposes)")
    parser.add_argument("--ddpm-checkpoint", type=str, help="Path to DDPM checkpoint directory")
    parser.add_argument("--classifier-ckpt", type=str, default=str(CHECKPOINTS_DIR / "mnist_classifier.pth"), help="Path to MNIST classifier for FID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-fid-samples", type=int, default=10000, help="Number of samples for FID calculation")
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
        [f"{vae_checkpoint_dir}/metrics.csv"],
        labels=["VAE"],
        save_path=f"{vae_checkpoint_dir}/training_curves.png",
    )

    plot_training_curves(
        [f"{diffusion_checkpoint_dir}/metrics.csv"],
        labels=["Latent DDPM"],
        save_path=f"{diffusion_checkpoint_dir}/training_curves.png",
    )

    # ===Generate samples===
    vae.eval()
    diffusion_model.eval()

    with torch.no_grad():
        latent_samples = diffusion_model.sample(4).cpu()
        dist = vae.decoder(latent_samples)

        # Get mean and noisy samples from the decoder distribution
        mean_samples = dist.mean.view(-1, 28, 28)
        noisy_samples = dist.sample().view(-1, 28, 28)

        # Rescale from [-1, 1] to [0, 1] for visualization
        mean_samples = ((mean_samples + 1) / 2).clamp(0, 1).cpu().numpy()
        noisy_samples = ((noisy_samples + 1) / 2).clamp(0, 1).cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for i in range(4):
        ax_mean = axes[0, i]
        ax_noisy = axes[1, i]

        ax_mean.imshow(mean_samples[i], cmap="gray", vmin=0, vmax=1)
        ax_mean.axis("off")

        ax_noisy.imshow(noisy_samples[i], cmap="gray", vmin=0, vmax=1)
        ax_noisy.axis("off")

    axes[0, 0].set_ylabel("Mean", fontsize=10)
    axes[1, 0].set_ylabel("Noisy", fontsize=10)

    fig.suptitle(f"Latent DDPM Samples: Mean vs Noisy (beta={args.beta})", fontsize=14)
    plt.tight_layout()

    save_path = f"{diffusion_checkpoint_dir}/generated_samples_mean_vs_noisy_beta{args.beta}.png"
    fig.savefig(save_path, dpi=600)
    plt.close(fig)
    print(f"Generated samples saved to {save_path}")

    #  ===FID Calculation ===
    logger.info(f"Computing FID with {args.n_fid_samples} samples...")

    # Load real test data
    test_data = get_dequantized_mnist(train=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.n_fid_samples, shuffle=True)
    x_real = next(iter(test_loader))[0].view(-1, 1, 28, 28).to(args.device)

    vae.to(args.device)
    diffusion_model.to(args.device)

    with torch.no_grad():
        z_ddpm = diffusion_model.sample(args.n_fid_samples)
        x_ddpm = vae.decoder(z_ddpm).mean.view(-1, 1, 28, 28)

    fid_ddpm = compute_fid(x_real, x_ddpm, device=args.device, classifier_ckpt=args.classifier_ckpt)

    logger.info(f"FID (Latent DDPM): {fid_ddpm:.2f}")

    #  ===Time Calculation for Sampling===
    import time

    logger.info("Measuring sampling speed...")

    n_samples = args.n_fid_samples

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        z_ddpm = diffusion_model.sample(n_samples)

    torch.cuda.synchronize()  # wait for GPU to finish
    end_time = time.time()

    elapsed_time = end_time - start_time
    samples_per_sec = n_samples / elapsed_time

    logger.info(f"Sampling time: {elapsed_time:.4f} seconds")
    logger.info(f"Sampling speed: {samples_per_sec:.2f} samples/sec")

    # Save FID rand time results to a text file
    fid_path = f"{diffusion_checkpoint_dir}/fid_beta{args.beta}.txt"
    with open(fid_path, "w") as f:
        f.write(f"beta={args.beta}\n")
        f.write(f"FID (Latent DDPM): {fid_ddpm:.2f}\n")
        f.write(f"Sampling time: {elapsed_time:.4f} seconds\n")
        f.write(f"Sampling speed: {samples_per_sec:.2f} samples/sec\n")
    logger.info(f"FID results saved to {fid_path}")

    logger.success("All visualizations complete!")
