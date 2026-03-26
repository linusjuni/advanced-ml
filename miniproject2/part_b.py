import os
import argparse
import math

import torch

from src.data import load_mnist
from src.model import build_ensemble_vae
from src.train import train
from src.geodesics import compute_ensemble_geodesic, ensemble_curve_length
from src.plotting import plot_latent_space_with_geodesics

from utils.logger import get_logger
from utils.settings import settings

logger = get_logger(__name__)


def build_model(latent_dim, num_decoders, device):
    return build_ensemble_vae(latent_dim, num_decoders).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiments/part_b",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoder ensemble members (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=25,
        metavar="N",
        help="number of geodesic pairs to compute (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",
        type=int,
        default=10,
        metavar="N",
        help="number of points along each geodesic curve (default: %(default)s)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=8,
        metavar="N",
        help="Monte Carlo samples per segment for ensemble energy (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.RANDOM_SEED,
        metavar="N",
        help="random seed (default: %(default)s)",
    )

    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        logger.info(f"{key} = {value}")

    torch.manual_seed(args.seed)

    device = args.device
    train_loader, test_loader = load_mnist(args.batch_size)

    if args.mode == "train":
        os.makedirs(args.experiment_folder, exist_ok=True)
        model = build_model(args.latent_dim, args.num_decoders, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_loader, args.epochs, device)
        save_path = f"{args.experiment_folder}/model.pt"
        torch.save(model.state_dict(), save_path)
        logger.success(f"Model saved to {save_path}")

    elif args.mode == "eval":
        model = build_model(args.latent_dim, args.num_decoders, device)
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        logger.info(f"Model loaded from {args.experiment_folder}/model.pt")
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, _ in test_loader:
                elbos.append(model.elbo(x.to(device)))

        logger.info(f"Mean test ELBO: {torch.tensor(elbos).mean().item():.4f}")

    elif args.mode == "geodesics":
        model = build_model(args.latent_dim, args.num_decoders, device)
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        logger.info(f"Model loaded from {args.experiment_folder}/model.pt")
        model.eval()

        zs, ys = [], []
        with torch.no_grad():
            for x, y in test_loader:
                zs.append(model.encoder(x.to(device)).mean.cpu())
                ys.append(y)

        zs = torch.cat(zs).numpy()
        ys = torch.cat(ys).numpy()

        rng = torch.Generator().manual_seed(42)
        indices = torch.randint(
            0, zs.shape[0], (args.num_curves, 2), generator=rng
        ).tolist()

        geodesic_curves = []
        distance_lines = []
        n = len(indices)
        for i, (start_idx, end_idx) in enumerate(indices):
            logger.info(f"Computing geodesic {i + 1}/{n} ({start_idx} -> {end_idx})")
            x_start = torch.tensor(zs[start_idx], dtype=torch.float32, device=device)
            x_end = torch.tensor(zs[end_idx], dtype=torch.float32, device=device)
            curve = compute_ensemble_geodesic(
                x_start,
                x_end,
                model.decoder,
                n_points=args.num_t,
                mc_samples=args.mc_samples,
            )
            geodesic_curves.append(curve.detach().cpu().numpy())

            geodesic_dist = ensemble_curve_length(
                curve,
                model.decoder,
                mc_samples=args.mc_samples,
            )
            euclidean_dist = math.dist(zs[start_idx].tolist(), zs[end_idx].tolist())
            distance_lines.append(
                (
                    f"Points {start_idx} and {end_idx}: "
                    f"euclidean={euclidean_dist:.4f}, geodesic={geodesic_dist:.4f}"
                )
            )

        os.makedirs(args.experiment_folder, exist_ok=True)
        distances_path = f"{args.experiment_folder}/distances.txt"
        with open(distances_path, "w") as f:
            f.write("\n".join(distance_lines) + "\n")
        logger.success(f"Distances saved to {distances_path}")

        plot_path = f"{args.experiment_folder}/latent_space.png"
        fig = plot_latent_space_with_geodesics(
            zs, ys, geodesic_curves, save_path=plot_path
        )
        logger.success(f"Plot saved to {plot_path}")
        fig.show()
