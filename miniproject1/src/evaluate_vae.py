import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributions as td

from src.dataset import get_binarized_mnist
from src.utils.logger import get_logger
from src.utils.model_utils import CHECKPOINTS_DIR, load_model
from src.utils.viz_utils import plot_prior_and_aggregate_posterior, plot_sample_grid
from src.vae import VAE, KLMode

logger = get_logger(__name__)

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


def compute_test_elbo(model: VAE, test_loader, device: str) -> tuple[float, float, float]:
    """Return mean (recon, kl, elbo) per sample over the full test set."""
    model.eval()
    model.to(device)
    total_recon, total_kl, total_elbo, n_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, *_ in test_loader:
            x = x.to(device)
            q, z = model.latent_representation(x)
            if model.kl_mode == KLMode.ANALYTIC:
                kl = td.kl_divergence(q, model.prior())
            else:
                kl = q.log_prob(z) - model.prior.log_prob(z)
            recon = model.decoder(z).log_prob(x)
            total_recon += recon.mean().item()
            total_kl += kl.mean().item()
            total_elbo += (recon - model.beta * kl).mean().item()
            n_batches += 1
    return total_recon / n_batches, total_kl / n_batches, total_elbo / n_batches


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

    summary: list[tuple[str, int, float, float, float, float, float, float]] = []

    for prior_name, pattern in PRIOR_PATTERNS.items():
        checkpoints = sorted(CHECKPOINTS_DIR.glob(pattern), key=_get_seed)
        if not checkpoints:
            logger.warning(f"No checkpoints found for prior={prior_name}")
            continue

        logger.info(f"{prior_name.upper()} prior ({len(checkpoints)} seeds)")
        recons, kls, elbos = [], [], []
        seed1_model = None

        for ckpt_dir in checkpoints:
            seed = _get_seed(ckpt_dir)
            model, _ = load_model(ckpt_dir)
            assert(isinstance(model, VAE))
            recon, kl, elbo = compute_test_elbo(model, test_loader, args.device)
            recons.append(recon)
            kls.append(kl)
            elbos.append(elbo)
            logger.info(f"seed={seed:2d}  recon={recon:.4f}  kl={kl:.4f}  elbo={elbo:.4f}")
            if seed == 1:
                seed1_model = model

        mean_recon, std_recon = float(np.mean(recons)), float(np.std(recons))
        mean_kl,   std_kl   = float(np.mean(kls)),   float(np.std(kls))
        mean_elbo, std_elbo = float(np.mean(elbos)),  float(np.std(elbos))
        logger.info(f"recon = {mean_recon:.4f} ± {std_recon:.4f}")
        logger.info(f"kl    = {mean_kl:.4f} ± {std_kl:.4f}")
        logger.info(f"elbo  = {mean_elbo:.4f} ± {std_elbo:.4f}")
        summary.append((prior_name, len(elbos),
                        mean_recon, std_recon,
                        mean_kl,   std_kl,
                        mean_elbo, std_elbo))

        if seed1_model is None:
            logger.warning(f"seed=1 checkpoint not found for {prior_name} — skipping plots")
            continue

        # Sample grid
        assert isinstance(seed1_model, VAE)
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
        writer.writerow(["prior", "n_seeds",
                         "mean_recon", "std_recon",
                         "mean_kl", "std_kl",
                         "mean_elbo", "std_elbo"])
        for row in summary:
            writer.writerow([row[0], row[1],
                             f"{row[2]:.4f}", f"{row[3]:.4f}",
                             f"{row[4]:.4f}", f"{row[5]:.4f}",
                             f"{row[6]:.4f}", f"{row[7]:.4f}"])

    # Summary table
    logger.info("Summary of test ELBO results:")
    logger.info(f"{'Prior':<12} {'Seeds':>5} {'Recon':>10} {'KL':>10} {'ELBO':>10}")
    for prior_name, n, mr, sr, mk, sk, me, se in summary:
        logger.info(f"{prior_name:<12} {n:>5} {mr:>10.4f} {mk:>10.4f} {me:>10.4f}")
    logger.success(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
