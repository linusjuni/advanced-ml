import argparse
import csv

import torch

from src.dataset import get_binarized_mnist
from src.vae import train
from src.utils.logger import get_logger
from src.utils.model_utils import (
    PriorType,
    VAEGaussianConfig,
    VAEMoGConfig,
    VAEFlowConfig,
    _build_vae,
    save_model,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on binarized MNIST")

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--seed", type=int, required=True, help="random seed")
    shared.add_argument("--M", type=int, required=True, help="latent dimension")
    shared.add_argument("--epochs", type=int, required=True, help="training epochs")
    shared.add_argument("--batch-size", type=int, required=True, help="batch size")
    shared.add_argument("--lr", type=float, required=True, help="learning rate")
    shared.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "cuda", "mps"],
        help="torch device",
    )
    subparsers = parser.add_subparsers(dest="prior", required=True)

    # Gaussian prior
    subparsers.add_parser("gaussian", parents=[shared], help="standard Gaussian prior")

    # MoG prior
    mog = subparsers.add_parser(
        "mog", parents=[shared], help="mixture of Gaussians prior"
    )
    mog.add_argument(
        "--K", type=int, required=True, help="number of mixture components"
    )

    # Flow prior
    flow = subparsers.add_parser("flow", parents=[shared], help="flow-based prior")
    flow.add_argument(
        "--flow-num-layers", type=int, required=True, help="number of flow layers"
    )
    flow.add_argument(
        "--flow-num-hidden", type=int, required=True, help="hidden units per flow layer"
    )

    return parser.parse_args()


def build_config(
    args: argparse.Namespace,
) -> VAEGaussianConfig | VAEMoGConfig | VAEFlowConfig:
    prior = PriorType(args.prior)
    match prior:
        case PriorType.GAUSSIAN:
            return VAEGaussianConfig(seed=args.seed, M=args.M)
        case PriorType.MOG:
            return VAEMoGConfig(seed=args.seed, M=args.M, K=args.K)
        case PriorType.FLOW:
            return VAEFlowConfig(
                seed=args.seed,
                M=args.M,
                flow_num_layers=args.flow_num_layers,
                flow_num_hidden=args.flow_num_hidden,
            )


def main():
    args = parse_args()

    logger.info("Training VAE with the following configuration:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key} = {value}")

    torch.manual_seed(args.seed)

    config = build_config(args)
    model = _build_vae(config).to(args.device)

    train_data = get_binarized_mnist(train=True)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_losses = train(model, optimizer, train_loader, args.epochs, args.device)

    run_dir = save_model(model, config)
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for epoch, loss in enumerate(epoch_losses, 1):
            writer.writerow([epoch, loss])

    logger.success(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
