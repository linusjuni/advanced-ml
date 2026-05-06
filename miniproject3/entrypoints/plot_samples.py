import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch

from src.data import load_mutag
from src.baseline import build_node_count_distribution
from src.evaluation import pyg_to_nx
from src.models.graphvae import GraphVAE
from utils.logger import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
NODE_COLOR = sns.color_palette("muted")[0]


def draw_graph(ax: plt.Axes, G: nx.Graph, node_colors: list | str, title: str) -> None:
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=80,
        width=0.8,
        edge_color=".5",
        with_labels=False,
    )
    ax.set_title(title, fontsize=7, pad=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real vs generated graph samples.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n", type=int, default=6, help="Number of graphs per row")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path (default: next to checkpoint)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output) if args.output else ckpt_path.parent / "sample_grid.pdf"

    # Load data — pick graphs spread across the node-count range for visual variety
    train_data, _ = load_mutag()
    sorted_indices = sorted(range(len(train_data)), key=lambda i: train_data[i].num_nodes)
    step = len(sorted_indices) // args.n
    indices = [sorted_indices[i * step] for i in range(args.n)]
    real_graphs = [pyg_to_nx(train_data[i]) for i in indices]

    # Load model and sample
    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GraphVAE(
        in_dim=ckpt["node_feature_dim"],
        hidden_dim=ckpt["args"]["hidden_dim"],
        latent_dim=ckpt["args"]["latent_dim"],
        max_nodes=ckpt["max_nodes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    node_counts, probs = build_node_count_distribution(train_data)
    gen_graphs = model.sample(args.n, device, node_counts, probs)
    logger.info("Sampled graphs", n=args.n)

    # Plot
    fig, axes = plt.subplots(2, args.n, figsize=(args.n * 1.8, 5))

    for col, G in enumerate(real_graphs):
        draw_graph(axes[0, col], G, NODE_COLOR, f"Real ({G.number_of_nodes()}n, {G.number_of_edges()}e)")

    for col, G in enumerate(gen_graphs):
        draw_graph(axes[1, col], G, NODE_COLOR, f"Gen ({G.number_of_nodes()}n, {G.number_of_edges()}e)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sample grid", path=str(out_path))


if __name__ == "__main__":
    main()
