import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(
    metrics_paths: list[str | Path],
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
):
    """Plot training loss curves from one or more metrics.csv files."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, path in enumerate(metrics_paths):
        path = Path(path)
        if not path.exists():
            continue
        epochs, losses = [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["train_loss"]))
        label = labels[i] if labels else path.parent.name
        ax.plot(epochs, losses, marker="o", linewidth=2, markersize=4, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (negative ELBO)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_sample_grid(
    samples: np.ndarray,
    n_rows: int = 8,
    n_cols: int = 8,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Plot a grid of 28x28 image samples.

    Parameters:
        samples: array of shape (N, 28, 28) or (N, 784).
    """
    samples = samples.reshape(-1, 28, 28)
    n_samples = min(len(samples), n_rows * n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(samples[i], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_prior_and_aggregate_posterior(
    model,
    data_loader,
    save_path: str | Path | None = None,
    n_prior_samples: int = 5000,
    title: str | None = None,
    device: str = "cpu",
):
    """Scatter plot of the aggregate posterior q(z) vs the prior p(z) for 2D latent space.

    Works for any prior type (Gaussian, MoG, Flow).

    Parameters:
        model: a VAE model (must have .encoder, .prior, and M=2).
        data_loader: DataLoader over the dataset to encode.
    """
    model.eval()
    model.to(device)

    # Collect aggregate posterior samples: encode every data point
    z_posterior = []
    with torch.no_grad():
        for x, *_ in data_loader:
            x = x.to(device)
            q = model.encoder(x)
            z = q.sample()
            z_posterior.append(z.cpu())
    z_posterior = torch.cat(z_posterior, dim=0).numpy()

    # Sample from the prior
    with torch.no_grad():
        z_prior = model.prior.sample(torch.Size([n_prior_samples])).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(z_prior[:, 0], z_prior[:, 1], alpha=0.3, s=4, c="tab:blue")
    axes[0].set_title("Prior p(z)")
    axes[0].set_xlabel("$z_1$")
    axes[0].set_ylabel("$z_2$")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(z_posterior[:, 0], z_posterior[:, 1], alpha=0.3, s=4, c="tab:orange")
    axes[1].set_title("Aggregate posterior q(z)")
    axes[1].set_xlabel("$z_1$")
    axes[1].set_ylabel("$z_2$")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
