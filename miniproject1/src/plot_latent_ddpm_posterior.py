"""
Plot β-VAE prior vs latent DDPM distribution vs aggregate posterior.

Usage:
    python -m src.plot_latent_ddpm_posterior \
        --vae-checkpoint <path> \
        --ddpm-checkpoint <path> \
        --beta 1e-6 \
        --save-path <output.png>
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.dataset import get_dequantized_mnist
from src.utils.model_utils import load_model
from src.utils.viz_utils import project_to_2d
from src.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="muted")

_GREYS_DARK = mcolors.LinearSegmentedColormap.from_list(
    "Greys_truncated", plt.get_cmap("Greys")(np.linspace(0.0, 0.75, 256))
)
_BLUES_DARK = mcolors.LinearSegmentedColormap.from_list(
    "Blues_truncated", plt.get_cmap("Blues")(np.linspace(0.2, 0.85, 256))
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--ddpm-checkpoint", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True, help="VAE β value (for title)")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-prior-samples", type=int, default=5000)
    parser.add_argument("--n-ddpm-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    save_path = args.save_path or str(Path(args.ddpm_checkpoint) / "posterior_comparison.png")

    logger.info("Loading β-VAE...")
    vae, _ = load_model(args.vae_checkpoint)
    vae.eval().to(args.device)

    logger.info("Loading latent DDPM...")
    ddpm, _ = load_model(args.ddpm_checkpoint)
    ddpm.eval().to(args.device)

    # --- Aggregate posterior: encode standard MNIST test set ---
    logger.info("Encoding test set for aggregate posterior...")
    test_data = get_dequantized_mnist(train=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    z_posterior, digit_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(args.device)
            z = vae.encoder(x).sample()
            z_posterior.append(z.cpu())
            digit_labels.append(y)
    z_posterior = torch.cat(z_posterior).numpy()
    digit_labels = torch.cat(digit_labels).numpy()

    # --- β-VAE prior samples (Gaussian N(0,I)) ---
    logger.info("Sampling from β-VAE prior...")
    with torch.no_grad():
        z_prior = vae.prior.sample(torch.Size([args.n_prior_samples])).cpu().numpy()

    # --- Latent DDPM samples ---
    logger.info("Sampling from latent DDPM...")
    with torch.no_grad():
        z_ddpm = ddpm.sample(args.n_ddpm_samples).cpu().numpy()

    # --- Project all three to 2D PCA (fit on aggregate posterior) ---
    logger.info("Projecting to 2D via PCA...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(z_posterior)
    z_post_2d = pca.transform(z_posterior)
    z_prior_2d = pca.transform(z_prior)
    z_ddpm_2d = pca.transform(z_ddpm)
    explained = pca.explained_variance_ratio_

    xlabel = f"PC1 ({100 * explained[0]:.1f}%)"
    ylabel = f"PC2 ({100 * explained[1]:.1f}%)"

    # Shared axis limits
    all_pts = np.concatenate([z_post_2d, z_prior_2d, z_ddpm_2d])
    x_margin = (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.05
    y_margin = (all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.05
    xlim = (all_pts[:, 0].min() - x_margin, all_pts[:, 0].max() + x_margin)
    ylim = (all_pts[:, 1].min() - y_margin, all_pts[:, 1].max() + y_margin)

    # --- Plot ---
    rng = np.random.default_rng(0)
    n_plot = 300  # posterior points per digit class

    fig, ax = plt.subplots(figsize=(7, 6))

    # β-VAE prior: grey KDE (same as Part A)
    sns.kdeplot(
        x=z_prior_2d[:, 0], y=z_prior_2d[:, 1],
        ax=ax, fill=True, cmap=_GREYS_DARK, levels=8,
        label=r"$\beta$-VAE prior $p(\mathbf{z})$",
    )

    # Latent DDPM distribution: blue KDE on top
    sns.kdeplot(
        x=z_ddpm_2d[:, 0], y=z_ddpm_2d[:, 1],
        ax=ax, fill=True, cmap=_BLUES_DARK, levels=8, alpha=0.55,
        label="Latent DDPM",
    )

    # Aggregate posterior: coloured scatter by digit class
    palette = sns.color_palette("muted", 10)
    for digit in range(10):
        mask = np.where(digit_labels == digit)[0]
        idx = rng.choice(mask, size=min(n_plot, len(mask)), replace=False)
        ax.scatter(
            z_post_2d[idx, 0], z_post_2d[idx, 1],
            alpha=0.6, s=12, color=palette[digit], label=str(digit),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(
        markerscale=3, fontsize=8, ncol=4, loc="upper right",
        title=f"β={args.beta}", title_fontsize=8,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)
    logger.info(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
