import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted")


def plot_latent_space_with_geodesics(zs, ys, geodesics, save_path=None):
    """
    Plot latent space scatter with geodesic curves overlaid.

    Parameters:
    zs: [np.ndarray]
        Latent means, shape (N, 2).
    ys: [np.ndarray]
        Class labels, shape (N,).
    geodesics: [list of np.ndarray]
        List of curves, each shape (n_points, 2).
    save_path: [str or None]
        If provided, save the figure to this path.
    """
    palette = sns.color_palette("muted")

    fig, ax = plt.subplots(figsize=(7, 6))
    for label in sorted(set(ys.tolist())):
        mask = ys == label
        ax.scatter(
            zs[mask, 0],
            zs[mask, 1],
            s=8,
            alpha=0.7,
            color=palette[int(label)],
            label=str(label),
        )

    for curve in geodesics:
        ax.plot(curve[:, 0], curve[:, 1], "-", color=palette[3], alpha=0.5, linewidth=1.5)
        ax.scatter(*curve[0], color=palette[2], s=50, zorder=5)
        ax.scatter(*curve[-1], color=palette[3], s=50, zorder=5)

    ax.legend(title="Class", markerscale=2)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("VAE Latent Space")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cov_vs_decoders(n_decoders, euc_means, euc_stds, geo_means, geo_stds, save_path=None):
    """
    Plot mean CoV ± std across point pairs as a function of ensemble size.

    Parameters
    ----------
    n_decoders : list[int]
        Number of ensemble decoders for each configuration.
    euc_means : array-like
        Mean per-pair CoV of Euclidean distances, one value per configuration.
    euc_stds : array-like
        Std of per-pair CoV of Euclidean distances, one value per configuration.
    geo_means : array-like
        Mean per-pair CoV of geodesic distances, one value per configuration.
    geo_stds : array-like
        Std of per-pair CoV of geodesic distances, one value per configuration.
    save_path : str or None
        If provided, save the figure to this path.
    """
    palette = sns.color_palette("muted")
    x = np.array(n_decoders)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.errorbar(
        x, euc_means, yerr=euc_stds,
        marker="o", markersize=6, linewidth=1.8, capsize=4, capthick=1.4,
        color=palette[0], markerfacecolor="white", markeredgewidth=1.8,
        label="Euclidean",
    )
    ax.errorbar(
        x, geo_means, yerr=geo_stds,
        marker="s", markersize=6, linewidth=1.8, capsize=4, capthick=1.4,
        color=palette[3], markerfacecolor="white", markeredgewidth=1.8,
        label="Geodesic",
    )

    ax.set_xlabel("Number of ensemble decoders", fontsize=11)
    ax.set_ylabel("Coefficient of Variation (CoV)", fontsize=11)
    ax.set_xticks(x)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
