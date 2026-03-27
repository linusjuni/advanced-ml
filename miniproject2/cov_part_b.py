"""
Analyse the reliability of geodesic vs. Euclidean distances as a function of
the number of ensemble decoders (Part B).

"""
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams


def parse_distances(path: Path) -> list[tuple[str, float, float]]:
    """Return (pair_key, euclidean, geodesic) tuples from a distances.txt file."""
    pattern = re.compile(
        r"Points (\d+) and (\d+): euclidean=([\d.]+), geodesic=([\d.]+)"
    )
    results = []
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            p1, p2, euc, geo = m.groups()
            results.append((f"{p1}-{p2}", float(euc), float(geo)))
    return results


def collect_distances(base: Path, models: list[str], seeds: list[int]):
    """Collect distances for all (model, seed) combinations."""
    data = {m: defaultdict(lambda: {"euclidean": [], "geodesic": []}) for m in models}
    for model in models:
        for seed in seeds:
            f = base / f"{model}_seed{seed}" / "distances.txt"
            if not f.exists():
                print(f"  [warn] missing: {f}")
                continue
            for key, euc, geo in parse_distances(f):
                data[model][key]["euclidean"].append(euc)
                data[model][key]["geodesic"].append(geo)
    return data


def per_pair_covs(data: dict, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pair CoV across seeds for a single model."""
    euc_covs, geo_covs = [], []
    for vals in data[model].values():
        e = np.array(vals["euclidean"])
        g = np.array(vals["geodesic"])
        euc_covs.append(e.std() / e.mean())
        geo_covs.append(g.std() / g.mean())
    return np.array(euc_covs), np.array(geo_covs)


def save_results(data: dict, models: list[str], n_decoders: list[int], save_path: Path):
    """Write results to CSV and print to terminal."""
    rows = []
    all_euc_covs, all_geo_covs = [], []

    def cov(x):
        return x.std() / x.mean()

    print("=" * 70)
    print(f"{'Model':<6} {'Decoders':<10} {'Dist Type':<12} {'Mean':>8} {'Std':>8} {'CoV':>8} {'Per-pair CoV':>14}")
    print("=" * 70)

    for model, nd in zip(models, n_decoders):
        euc_covs, geo_covs = per_pair_covs(data, model)
        all_euc = np.concatenate([v["euclidean"] for v in data[model].values()])
        all_geo = np.concatenate([v["geodesic"] for v in data[model].values()])

        rows.append({
            "model": model,
            "n_decoders": nd,
            "dist_type": "euclidean",
            "mean": round(float(all_euc.mean()), 4),
            "std": round(float(all_euc.std()), 4),
            "cov_pooled": round(float(cov(all_euc)), 4),
            "cov_per_pair_mean": round(float(euc_covs.mean()), 4),
            "cov_per_pair_std": round(float(euc_covs.std()), 4),
        })
        rows.append({
            "model": model,
            "n_decoders": nd,
            "dist_type": "geodesic",
            "mean": round(float(all_geo.mean()), 4),
            "std": round(float(all_geo.std()), 4),
            "cov_pooled": round(float(cov(all_geo)), 4),
            "cov_per_pair_mean": round(float(geo_covs.mean()), 4),
            "cov_per_pair_std": round(float(geo_covs.std()), 4),
        })

        print(
            f"{model:<6} {nd:<10} {'euclidean':<12}"
            f" {all_euc.mean():>8.4f} {all_euc.std():>8.4f} {cov(all_euc):>8.4f}"
            f" {euc_covs.mean():>8.4f} ± {euc_covs.std():.4f}"
        )
        print(
            f"{model:<6} {nd:<10} {'geodesic':<12}"
            f" {all_geo.mean():>8.4f} {all_geo.std():>8.4f} {cov(all_geo):>8.4f}"
            f" {geo_covs.mean():>8.4f} ± {geo_covs.std():.4f}"
        )

        all_euc_covs.append(euc_covs)
        all_geo_covs.append(geo_covs)

    print("=" * 70)
    flat_euc = np.concatenate(all_euc_covs)
    flat_geo = np.concatenate(all_geo_covs)
    print("Per-pair CoV across ALL models (mean ± std):")
    print(f"  Euclidean CoV: {flat_euc.mean():.4f} ± {flat_euc.std():.4f}")
    print(f"  Geodesic  CoV: {flat_geo.mean():.4f} ± {flat_geo.std():.4f}")
    print()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        f.write("\n# Summary across all models\n")
        f.write(f"# euclidean_cov_mean: {flat_euc.mean():.4f}\n")
        f.write(f"# euclidean_cov_std: {flat_euc.std():.4f}\n")
        f.write(f"# geodesic_cov_mean: {flat_geo.mean():.4f}\n")
        f.write(f"# geodesic_cov_std: {flat_geo.std():.4f}\n")

    print(f"Results saved to {save_path}")


def plot_cov_vs_decoders(data: dict, models: list[str], n_decoders: list[int], save_path: str = None):
    """Plot CoV vs. number of decoders."""
    rcParams["font.family"] = "sans-serif"
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False

    euc_means, euc_stds = [], []
    geo_means, geo_stds = [], []

    for model in models:
        euc_covs, geo_covs = per_pair_covs(data, model)
        euc_means.append(euc_covs.mean())
        euc_stds.append(euc_covs.std())
        geo_means.append(geo_covs.mean())
        geo_stds.append(geo_covs.std())

    x = np.array(n_decoders)
    C_EUC = "#c0392b"
    C_GEO = "#2471a3"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    ax.errorbar(
        x, euc_means, yerr=euc_stds,
        marker="o", markersize=7, linewidth=2, capsize=4, capthick=1.5,
        color=C_EUC, markerfacecolor="white", markeredgewidth=2,
        label="Euclidean",
    )
    ax.errorbar(
        x, geo_means, yerr=geo_stds,
        marker="s", markersize=7, linewidth=2, capsize=4, capthick=1.5,
        color=C_GEO, markerfacecolor="white", markeredgewidth=2,
        label="Geodesic",
    )

    ax.set_xlabel("Number of ensemble decoders", fontsize=12)
    ax.set_ylabel("CoV", fontsize=12)
    ax.set_xticks(x)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color="lightgrey", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11, loc="upper right", bbox_to_anchor=(0.95, 0.93))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    return fig


if __name__ == "__main__":
    base = Path("experiments/linus/part_b")
    models = ["m1", "m2", "m3"]
    n_decoders = [1, 2, 3]
    seeds = list(range(1, 11))

    data = collect_distances(base, models, seeds)
    save_results(data, models, n_decoders, Path("results/cov_results.csv"))
    plot_cov_vs_decoders(data, models, n_decoders, "results/cov_vs_decoders.png")