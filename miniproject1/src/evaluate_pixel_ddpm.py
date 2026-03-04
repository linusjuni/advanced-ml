import argparse
import time

import matplotlib.pyplot as plt
import torch

from src.ddpm import DDPM
from src.dataset import get_dequantized_mnist
from src.fid import compute_fid
from src.utils.model_utils import load_model, CHECKPOINTS_DIR
from src.utils.viz_utils import plot_training_curves
from src.utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pixel-space DDPM on MNIST")
    parser.add_argument("--ddpm-checkpoint", type=str, required=True, help="Path to DDPM checkpoint directory")
    parser.add_argument("--classifier-ckpt", type=str, default=str(CHECKPOINTS_DIR / "mnist_classifier.pth"), help="Path to MNIST classifier for FID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-fid-samples", type=int, default=10000, help="Number of samples for FID calculation")
    args = parser.parse_args()

    ckpt_dir = args.ddpm_checkpoint

    logger.info("Loading pixel-space DDPM...")
    model, config = load_model(ckpt_dir)
    assert isinstance(model, DDPM)
    logger.info(f"Loaded DDPM with config: {config}")

    # Training curve
    plot_training_curves(
        [f"{ckpt_dir}/metrics.csv"],
        labels=["Pixel DDPM"],
        save_path=f"{ckpt_dir}/training_curves.png",
    )

    model.eval()
    model.to(args.device)

    # === Generate 4 samples ===
    with torch.no_grad():
        samples = model.sample(4)

    samples = samples.view(-1, 28, 28)
    samples = ((samples + 1) / 2).clamp(0, 1).cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("Pixel DDPM samples", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{ckpt_dir}/generated_samples.png", dpi=600)
    plt.close(fig)
    logger.info(f"Sample grid saved to {ckpt_dir}/generated_samples.png")

    # === FID ===
    logger.info(f"Computing FID with {args.n_fid_samples} samples...")

    test_data = get_dequantized_mnist(train=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.n_fid_samples, shuffle=True)
    x_real = next(iter(test_loader))[0].view(-1, 1, 28, 28).to(args.device)

    with torch.no_grad():
        x_gen = model.sample(args.n_fid_samples)
    x_gen = x_gen.view(-1, 1, 28, 28)

    fid = compute_fid(x_real, x_gen, device=args.device, classifier_ckpt=args.classifier_ckpt)
    logger.info(f"FID (Pixel DDPM): {fid:.2f}")

    # === Sampling time ===
    logger.info("Measuring sampling speed...")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        model.sample(args.n_fid_samples)

    torch.cuda.synchronize()
    elapsed = time.time() - start
    samples_per_sec = args.n_fid_samples / elapsed

    logger.info(f"Sampling time: {elapsed:.4f} seconds")
    logger.info(f"Sampling speed: {samples_per_sec:.2f} samples/sec")

    # === Save results ===
    fid_path = f"{ckpt_dir}/fid.txt"
    with open(fid_path, "w") as f:
        f.write(f"FID (Pixel DDPM): {fid:.2f}\n")
        f.write(f"Sampling time: {elapsed:.4f} seconds\n")
        f.write(f"Sampling speed: {samples_per_sec:.2f} samples/sec\n")
    logger.info(f"Results saved to {fid_path}")

    logger.success("Evaluation complete!")
