"""
evaluate_vae.py — Part A: evaluate trained VAE models.

Outputs (saved to src/results/part_a/):
  test_elbo.csv          mean/std test ELBO per prior over all seeds
  samples_<prior>.png    8×8 sample grid from the seed-1 model
  posterior_<prior>.png  prior vs aggregate posterior scatter (PCA, overlaid)

Usage:
  python -m src.evaluate_vae [--device cuda] [--batch-size 256]
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch

from src.dataset import get_binarized_mnist
from src.utils.model_utils import CHECKPOINTS_DIR, load_model
from src.utils.viz_utils import plot_prior_and_aggregate_posterior, plot_sample_grid

RESULTS_DIR = Path(__file__).parent / "results" / "part_a"

# Glob patterns that match checkpoints produced by the Part A training runs.
PRIOR_PATTERNS: dict[str, str] = {
    "gaussian": "vae_M32_decoder_typebernoulli_beta1.0_priorgaussian_seed*",
    "mog":      "vae_M32_decoder_typebernoulli_beta1.0_K10_priormog_seed*",
    "flow":     "vae_M32_decoder_typebernoulli_beta1.0_flow_*_priorflow_seed*",
}


def _get_seed(path: Path) -> int:
    m = re.search(r"seed(\d+)", path.name)
    return int(m.group(1)) if m else -1


def compute_test_elbo(model, test_loader, device: str) -> float:
    """Return the mean test ELBO (per sample) over the full test set."""
    model.eval()
    model.to(device)
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for x, *_ in test_loader:
            x = x.to(device)
            total += (-model(x)).item()   # model.forward returns −ELBO
            n_batches += 1
    return total / n_batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Part A VAE models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_data = get_binarized_mnist(train=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    summary: list[tuple[str, int, float, float]] = []

    for prior_name, pattern in PRIOR_PATTERNS.items():
        checkpoints = sorted(CHECKPOINTS_DIR.glob(pattern), key=_get_seed)
        if not checkpoints:
            print(f"[WARN] No checkpoints found for prior={prior_name}")
            continue

        print(f"\n=== {prior_name.upper()} prior ({len(checkpoints)} seeds) ===")
        elbos: list[float] = []
        seed1_model = None

        for ckpt_dir in checkpoints:
            seed = _get_seed(ckpt_dir)
            model, _ = load_model(ckpt_dir)
            elbo = compute_test_elbo(model, test_loader, args.device)
            elbos.append(elbo)
            print(f"  seed={seed:2d}  test ELBO = {elbo:.4f}")
            if seed == 1:
                seed1_model = model

        mean_elbo = float(np.mean(elbos))
        std_elbo = float(np.std(elbos))
        print(f"  → mean = {mean_elbo:.4f},  std = {std_elbo:.4f}")
        summary.append((prior_name, len(elbos), mean_elbo, std_elbo))

        if seed1_model is None:
            print(f"  [WARN] seed=1 checkpoint not found for {prior_name} — skipping plots")
            continue

        # Sample grid
        seed1_model.eval().to(args.device)
        with torch.no_grad():
            samples = seed1_model.sample(64).cpu().numpy()
        plot_sample_grid(
            samples,
            title=f"VAE samples — {prior_name} prior (seed 1)",
            save_path=RESULTS_DIR / f"samples_{prior_name}.png",
        )

        # Prior vs aggregate posterior
        plot_prior_and_aggregate_posterior(
            model=seed1_model,
            data_loader=test_loader,
            title=f"{prior_name.upper()} prior vs aggregate posterior",
            save_path=RESULTS_DIR / f"posterior_{prior_name}.png",
            device=args.device,
        )

    # Save CSV
    csv_path = RESULTS_DIR / "test_elbo.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prior", "n_seeds", "mean_elbo", "std_elbo"])
        for row in summary:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}"])

    # Summary table
    print("\n" + "=" * 48)
    print(f"{'Prior':<12} {'Seeds':>5} {'Mean ELBO':>12} {'Std':>8}")
    print("-" * 48)
    for prior_name, n, mean, std in summary:
        print(f"{prior_name:<12} {n:>5} {mean:>12.4f} {std:>8.4f}")
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
