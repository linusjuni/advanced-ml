# Train pixel-space DDPM with U-Net on standard MNIST (Part B)

import argparse

import torch

from src.dataset import get_dequantized_mnist
from src.ddpm import train
from src.utils.logger import get_logger
from src.utils.model_utils import (
    DDPMConfig,
    _build_ddpm,
    make_run_dir,
    save_model,
    save_metrics,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM with U-Net on MNIST")
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    parser.add_argument("--epochs", type=int, required=True, help="training epochs")
    parser.add_argument("--batch-size", type=int, required=True, help="batch size")
    parser.add_argument("--lr", type=float, required=True, help="learning rate")
    parser.add_argument("--T", type=int, default=1000, help="number of diffusion steps")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "cuda", "mps"],
        help="torch device",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Training DDPM with U-Net:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key} = {value}")

    torch.manual_seed(args.seed)

    config = DDPMConfig(T=args.T)
    model = _build_ddpm(config).to(args.device)

    train_data = get_dequantized_mnist(train=True)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_losses = train(model, optimizer, train_loader, args.epochs, args.device)

    run_dir = make_run_dir(
        config,
        seed=args.seed,
        lr=args.lr,
        bs=args.batch_size,
        ep=args.epochs,
    )
    save_model(model, config, run_dir)
    save_metrics(run_dir, epoch_losses)

    logger.success(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()