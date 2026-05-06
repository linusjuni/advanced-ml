import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.baseline import ErdosRenyiBaseline, build_node_count_distribution
from src.data import load_mutag
from src.evaluation import (
    compute_graph_statistics,
    compute_novelty_uniqueness,
    pyg_to_nx,
)
from src.models.graphvae import GraphVAE
from utils.logger import get_logger
from utils.settings import settings

logger = get_logger(__name__)


def plot_histogram_grid(
    empirical: dict[str, list[float]],
    baseline: dict[str, list[float]],
    graphvae: dict[str, list[float]],
    save_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", palette="muted")

    stat_keys = ["node_degree", "clustering_coefficient", "eigenvector_centrality"]
    stat_labels = ["Node degree", "Clustering coefficient", "Eigenvector centrality"]
    sources = [("Empirical", empirical), ("Baseline", baseline), ("GraphVAE", graphvae)]
    colors = sns.color_palette("muted", n_colors=3)

    fig, axes = plt.subplots(3, 3, figsize=(10, 8))

    for row, (key, label) in enumerate(zip(stat_keys, stat_labels)):
        all_vals = empirical[key] + baseline[key] + graphvae[key]
        bins = np.linspace(min(all_vals), max(all_vals), 30)

        for col, (name, stats) in enumerate(sources):
            ax = axes[row, col]
            ax.hist(stats[key], bins=bins, density=True, color=colors[col], edgecolor="white", linewidth=0.5)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(label)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and GraphVAE on MUTAG.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(settings.OUTPUT_DIR) / "evaluation" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    train_data, _ = load_mutag()
    train_graphs = [pyg_to_nx(d) for d in train_data]
    logger.info("Converted training data to nx", num_graphs=len(train_graphs))

    # Baseline sampling
    baseline = ErdosRenyiBaseline(train_data)
    baseline_graphs = baseline.sample_many(args.num_samples)

    # GraphVAE sampling
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = GraphVAE(
        in_dim=ckpt["node_feature_dim"],
        hidden_dim=ckpt["args"]["hidden_dim"],
        latent_dim=ckpt["args"]["latent_dim"],
        max_nodes=ckpt["max_nodes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    node_counts, probs = build_node_count_distribution(train_data)
    graphvae_graphs = model.sample(args.num_samples, device, node_counts, probs)
    logger.success("Sampled from GraphVAE", num_samples=args.num_samples)

    # Novelty / uniqueness
    baseline_nu = compute_novelty_uniqueness(baseline_graphs, train_graphs)
    graphvae_nu = compute_novelty_uniqueness(graphvae_graphs, train_graphs)

    nu_table = {"baseline": baseline_nu, "graphvae": graphvae_nu}
    with open(run_dir / "novelty_uniqueness.json", "w") as f:
        json.dump(nu_table, f, indent=2)
    logger.info("Saved novelty/uniqueness", path=str(run_dir / "novelty_uniqueness.json"))

    # Graph statistics
    empirical_stats = compute_graph_statistics(train_graphs)
    baseline_stats = compute_graph_statistics(baseline_graphs)
    graphvae_stats = compute_graph_statistics(graphvae_graphs)

    stats_out = {"empirical": empirical_stats, "baseline": baseline_stats, "graphvae": graphvae_stats}
    with open(run_dir / "graph_statistics.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    logger.info("Saved graph statistics", path=str(run_dir / "graph_statistics.json"))

    # 3x3 histogram grid
    plot_histogram_grid(empirical_stats, baseline_stats, graphvae_stats, run_dir / "histogram_grid.pdf")
    logger.success("Saved histogram grid", path=str(run_dir / "histogram_grid.pdf"))

    logger.success("Done", run_dir=str(run_dir))


if __name__ == "__main__":
    main()
