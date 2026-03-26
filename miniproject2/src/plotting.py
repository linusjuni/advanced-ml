import matplotlib
matplotlib.use('Agg')
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
