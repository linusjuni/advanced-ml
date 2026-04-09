"""
Analyse the reliability of geodesic vs. Euclidean distances as a function of
the number of ensemble decoders (Part B).
"""

import re
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.plotting import plot_cov_vs_decoders


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
    """
    Compute per-pair CoV across seeds for a single model.

    CoV_ij = std({d_ij^(1), ..., d_ij^(M)}) / mean({d_ij^(1), ..., d_ij^(M)})

    Uses sample std (ddof=1) as per standard statistical convention.
    """
    euc_covs, geo_covs = [], []
    for vals in data[model].values():
        e = np.array(vals["euclidean"])
        g = np.array(vals["geodesic"])
        euc_covs.append(e.std(ddof=1) / e.mean())
        geo_covs.append(g.std(ddof=1) / g.mean())
    return np.array(euc_covs), np.array(geo_covs)


def save_results(data: dict, models: list[str], n_decoders: list[int], save_path: Path):
    """Write per-pair CoV statistics to CSV and print summary to terminal."""
    rows = []

    print("=" * 72)
    print(
        f"{'Model':<6} {'Decoders':<10} {'Dist Type':<12} {'CoV mean':>10} {'CoV std':>10} {'N pairs':>8}"
    )
    print("=" * 72)

    for model, nd in zip(models, n_decoders):
        euc_covs, geo_covs = per_pair_covs(data, model)
        n_pairs = len(euc_covs)

        rows.append(
            {
                "model": model,
                "n_decoders": nd,
                "dist_type": "euclidean",
                "n_pairs": n_pairs,
                "cov_mean": round(float(euc_covs.mean()), 4),
                "cov_std": round(float(euc_covs.std(ddof=1)), 4),
            }
        )
        rows.append(
            {
                "model": model,
                "n_decoders": nd,
                "dist_type": "geodesic",
                "n_pairs": n_pairs,
                "cov_mean": round(float(geo_covs.mean()), 4),
                "cov_std": round(float(geo_covs.std(ddof=1)), 4),
            }
        )

        print(
            f"{model:<6} {nd:<10} {'euclidean':<12} {euc_covs.mean():>10.4f} {euc_covs.std(ddof=1):>10.4f} {n_pairs:>8}"
        )
        print(
            f"{model:<6} {nd:<10} {'geodesic':<12} {geo_covs.mean():>10.4f} {geo_covs.std(ddof=1):>10.4f} {n_pairs:>8}"
        )

    print("=" * 72)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    base = Path("experiments/linus/part_b")
    models = ["m1", "m2", "m3"]
    n_decoders = [1, 2, 3]
    seeds = list(range(1, 11))

    data = collect_distances(base, models, seeds)
    save_results(data, models, n_decoders, Path("results/cov_results.csv"))

    euc_means, euc_stds, geo_means, geo_stds = [], [], [], []
    for model in models:
        euc_covs, geo_covs = per_pair_covs(data, model)
        euc_means.append(euc_covs.mean())
        euc_stds.append(euc_covs.std(ddof=1))
        geo_means.append(geo_covs.mean())
        geo_stds.append(geo_covs.std(ddof=1))

    plot_cov_vs_decoders(
        n_decoders,
        euc_means,
        euc_stds,
        geo_means,
        geo_stds,
        save_path="results/cov_vs_decoders.png",
    )
