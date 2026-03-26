import matplotlib.pyplot as plt


def plot_latent_space(zs, ys, geodesics, save_path=None):
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
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(zs[:, 0], zs[:, 1], c=ys, s=8, cmap="tab10", alpha=0.7)

    for curve in geodesics:
        ax.plot(curve[:, 0], curve[:, 1], "r-", alpha=0.5, linewidth=1)
        ax.scatter(*curve[0], c="green", s=50, zorder=5)
        ax.scatter(*curve[-1], c="red", s=50, zorder=5)

    plt.colorbar(sc, ax=ax)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("VAE Latent Space")

    if save_path:
        fig.savefig(save_path)
    return fig
