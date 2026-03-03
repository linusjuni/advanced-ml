import argparse

import torch

from src.dataset import get_binarized_mnist, get_dequantized_mnist
from src.vae import train
from src.utils.logger import get_logger
from src.utils.model_utils import (
    PriorType,
    DecoderType,
    VAEGaussianConfig,
    VAEMoGConfig,
    VAEFlowConfig,
    _build_vae,
    make_run_dir,
    save_model,
    save_metrics,
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
    shared.add_argument("--beta", type=float, default=1.0, help="beta parameter for beta-VAE (default: 1.0)")
    shared.add_argument(
        "--decoder-type",
        type=str,
        default="bernoulli",
        choices=["bernoulli", "gaussian"],
        help="decoder type: bernoulli for binary MNIST (Part A), gaussian for standard MNIST (Part B)"
    )
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
    decoder_type = DecoderType(args.decoder_type)
    beta = args.beta
    
    match prior:
        case PriorType.GAUSSIAN:
            return VAEGaussianConfig(M=args.M, decoder_type=decoder_type, beta=beta)
        case PriorType.MOG:
            return VAEMoGConfig(M=args.M, K=args.K, decoder_type=decoder_type, beta=beta)
        case PriorType.FLOW:
            return VAEFlowConfig(
                M=args.M,
                flow_num_layers=args.flow_num_layers,
                flow_num_hidden=args.flow_num_hidden,
                decoder_type=decoder_type,
                beta=beta,
            )


def main():
    args = parse_args()

    logger.info("Training VAE with the following configuration:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key} = {value}")

    torch.manual_seed(args.seed)

    config = build_config(args)
    model = _build_vae(config).to(args.device)

    # Choose dataset based on decoder type
    if config.decoder_type == DecoderType.GAUSSIAN:
        # Part B: Standard MNIST with Gaussian decoder
        train_data = get_dequantized_mnist(train=True)
    else:
        # Part A: Binary MNIST with Bernoulli decoder
        train_data = get_binarized_mnist(train=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
      = train(model, optimizer, train_loader, args.epochs, args.device)

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
