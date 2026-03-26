import os
import argparse

import torch

from src.data import load_mnist
from src.model import build_ensemble_vae
from src.train import train

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
        choices=["train", "eval"],
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
