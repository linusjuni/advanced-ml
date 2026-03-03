import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sns.set_theme(style="whitegrid", palette="muted")

# Greys truncated to [white → dark grey], avoiding pure black at peak density
_GREYS_DARK = mcolors.LinearSegmentedColormap.from_list(
    "Greys_truncated", plt.get_cmap("Greys")(np.linspace(0.0, 0.75, 256))
)


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
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600)
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
        fig.savefig(save_path, dpi=600)
    plt.close(fig)
    return fig


def project_to_2d(
    z_posterior: np.ndarray,
    z_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Project latent samples to 2D for visualisation.

    PCA is fit on ``z_posterior`` and applied to both arrays, so the axes
    reflect the directions of maximum variance in the aggregate posterior.
    For M=2 the data is returned unchanged.

    Parameters:
        z_posterior: (N, M) posterior samples used to fit PCA.
        z_other:     (K, M) other samples (e.g. prior) to project with the
                     same transformation.

    Returns:
        z_posterior_2d: (N, 2)
        z_other_2d:     (K, 2)
        explained_variance_ratio: (2,) or None when M=2.
    """
    from sklearn.decomposition import PCA

    M = z_posterior.shape[1]
    if M > 2:
        pca = PCA(n_components=2)
        pca.fit(z_posterior)
        return pca.transform(z_posterior), pca.transform(z_other), pca.explained_variance_ratio_
    return z_posterior, z_other, None


def plot_prior_and_aggregate_posterior(
    model,
    data_loader,
    save_path: str | Path | None = None,
    n_prior_samples: int = 5000,
    title: str | None = None,
    device: str = "cpu",
):
    """Overlaid scatter plot of the prior p(z) and aggregate posterior q(z).

    Posterior samples are coloured by MNIST digit class (0–9).  For M>2 the
    latent space is projected to 2D via PCA (fit on the posterior) before
    plotting.  Both distributions share the same axis limits so scale
    differences are immediately visible.

    Parameters:
        model:           VAE with .encoder and .prior.
        data_loader:     DataLoader returning (image, label) pairs.
        n_prior_samples: Number of samples drawn from the prior.
    """
    model.eval()
    model.to(device)

    # Aggregate posterior samples + digit labels
    z_posterior, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            z = model.encoder(x).sample()
            z_posterior.append(z.cpu())
            labels.append(y)
    z_posterior = torch.cat(z_posterior).numpy()
    labels = torch.cat(labels).numpy()

    # Prior samples
    with torch.no_grad():
        z_prior = model.prior.sample(torch.Size([n_prior_samples])).cpu().numpy()

    # Project both to 2D
    z_post_2d, z_prior_2d, explained = project_to_2d(z_posterior, z_prior)

    if explained is not None:
        xlabel = f"PC1 ({100 * explained[0]:.1f}%)"
        ylabel = f"PC2 ({100 * explained[1]:.1f}%)"
    else:
        xlabel, ylabel = "$z_1$", "$z_2$"

    # Shared axis limits with a small margin
    all_pts = np.concatenate([z_post_2d, z_prior_2d])
    x_margin = (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.05
    y_margin = (all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.05
    xlim = (all_pts[:, 0].min() - x_margin, all_pts[:, 0].max() + x_margin)
    ylim = (all_pts[:, 1].min() - y_margin, all_pts[:, 1].max() + y_margin)

    # Subsample posterior for legibility: at most n_plot points per class
    rng = np.random.default_rng(0)
    n_plot = 300

    fig, ax = plt.subplots(figsize=(7, 6))

    # Prior: filled KDE as background — darker grey = higher density
    sns.kdeplot(
        x=z_prior_2d[:, 0], y=z_prior_2d[:, 1],
        ax=ax, fill=True, cmap=_GREYS_DARK, levels=8, label="prior p(z)",
    )

    # Posterior coloured by digit class
    palette = sns.color_palette("muted", 10)
    for digit in range(10):
        mask = np.where(labels == digit)[0]
        idx = rng.choice(mask, size=min(n_plot, len(mask)), replace=False)
        ax.scatter(z_post_2d[idx, 0], z_post_2d[idx, 1],
                   alpha=0.6, s=12, color=palette[digit], label=str(digit))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(markerscale=3, fontsize=8, ncol=6, loc="upper right",
              title="digit / prior", title_fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600)
    plt.close(fig)
    return fig
