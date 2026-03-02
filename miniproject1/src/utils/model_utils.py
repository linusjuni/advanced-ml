import csv
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

from src.vae import VAE, GaussianEncoder, BernoulliDecoder, GaussianDecoder, KLMode
from src.priors import GaussianPrior, MoGPrior, FlowPrior
from src.flow import Flow, GaussianBase, MaskedCouplingLayer

CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"


class ModelType(str, Enum):
    VAE = "vae"
    DDPM = "ddpm"
    LATENT_DDPM = "latent_ddpm"


class PriorType(str, Enum):
    GAUSSIAN = "gaussian"
    MOG = "mog"
    FLOW = "flow"


class DecoderType(str, Enum):
    BERNOULLI = "bernoulli"  # For binary MNIST (Part A)
    GAUSSIAN = "gaussian"    # For standard MNIST (Part B)


@dataclass(kw_only=True)
class VAEBaseConfig:
    M: int
    decoder_type: DecoderType = DecoderType.BERNOULLI
    beta: float = 1.0  # β parameter for β-VAE
    model_type: ModelType = ModelType.VAE


@dataclass(kw_only=True)
class VAEGaussianConfig(VAEBaseConfig):
    prior: PriorType = PriorType.GAUSSIAN


@dataclass(kw_only=True)
class VAEMoGConfig(VAEBaseConfig):
    K: int
    prior: PriorType = PriorType.MOG


@dataclass(kw_only=True)
class VAEFlowConfig(VAEBaseConfig):
    flow_num_layers: int
    flow_num_hidden: int
    prior: PriorType = PriorType.FLOW


@dataclass(kw_only=True)
class DDPMBaseConfig:
    # TODO (Part B)
    model_type: ModelType = ModelType.DDPM


@dataclass(kw_only=True)
class DDPMConfig(DDPMBaseConfig):
    T: int = 1000
    beta_1: float = 1e-4
    beta_T: float = 2e-2


@dataclass(kw_only=True)
class LatentDDPMConfig(DDPMBaseConfig):
    M: int  # Latent dimension from VAE
    num_hidden: int = 512  # Hidden units for FcNetwork
    T: int = 1000
    beta_1: float = 1e-4
    beta_T: float = 2e-2
    model_type: ModelType = ModelType.LATENT_DDPM


ModelConfig = (
    VAEGaussianConfig | VAEMoGConfig | VAEFlowConfig | DDPMConfig | LatentDDPMConfig
)


def make_run_dir(config: ModelConfig, **training_params: object) -> Path:
    """
    Create a run directory name from model config and training params.

    Parameters:
    config: [ModelConfig]
        The model architecture config.
    **training_params:
        Training hyperparameters (seed, lr, bs, epochs, etc.).

    Returns:
    run_dir: [Path]
        The created run directory path.
    """
    parts = [config.model_type.value]
    for key, value in asdict(config).items():
        if key == "model_type":
            continue
        if isinstance(value, Enum):
            value = value.value
        parts.append(f"{key}{value}")
    for key, value in training_params.items():
        parts.append(f"{key}{value}")
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = CHECKPOINTS_DIR / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_model(model: nn.Module, config: ModelConfig, run_dir: Path) -> Path:
    """
    Save a model's weights and config to a run directory.

    Parameters:
    model: [nn.Module]
        The model to save.
    config: [ModelConfig]
        A dataclass describing the model architecture and hyperparameters.
    run_dir: [Path]
        Directory to save into (created by make_run_dir).

    Returns:
    run_dir: [Path]
        The run directory path.
    """
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(config)},
        run_dir / "model.pth",
    )
    return run_dir


def save_metrics(run_dir: Path, epoch_losses: list[float]) -> None:
    """
    Save per-epoch training metrics to a CSV file in the run directory.

    Parameters:
    run_dir: [Path]
        The run directory (created by make_run_dir).
    epoch_losses: [list[float]]
        Average training loss per epoch.
    """
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for epoch, loss in enumerate(epoch_losses, 1):
            writer.writerow([epoch, loss])


def load_model(path: str | Path) -> tuple[nn.Module, ModelConfig]:
    """
    Load a model from disk by reconstructing it from the saved config
    and loading its weights.

    Parameters:
    path: [str | Path]
        Path to a run directory (containing model.pth) or directly to a .pth file.

    Returns:
    model: [nn.Module]
        The reconstructed model with weights loaded, in eval mode.
    config: [ModelConfig]
        The config dataclass that was saved alongside the weights.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "model.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    model_type = config_dict["model_type"]

    if model_type == ModelType.VAE:
        prior = config_dict["prior"]
        match prior:
            case PriorType.GAUSSIAN:
                config = VAEGaussianConfig(**config_dict)
            case PriorType.MOG:
                config = VAEMoGConfig(**config_dict)
            case PriorType.FLOW:
                config = VAEFlowConfig(**config_dict)
            case _:
                raise ValueError(f"Unknown VAE prior: {prior!r}")
        model = _build_vae(config)

    elif model_type == ModelType.DDPM:
        config = DDPMConfig(**config_dict)
        model = _build_ddpm(config)

    elif model_type == ModelType.LATENT_DDPM:
        config = LatentDDPMConfig(**config_dict)
        model = _build_latent_ddpm(config)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def _build_vae(config: VAEGaussianConfig | VAEMoGConfig | VAEFlowConfig) -> nn.Module:

    M = config.M

    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M * 2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )

    if isinstance(config, VAEGaussianConfig):
        prior = GaussianPrior(M)
        kl_mode = KLMode.ANALYTIC

    elif isinstance(config, VAEMoGConfig):
        prior = MoGPrior(M, config.K)
        kl_mode = KLMode.MONTE_CARLO

    elif isinstance(config, VAEFlowConfig):
        prior = _build_flow_prior(M, config.flow_num_layers, config.flow_num_hidden)
        kl_mode = KLMode.MONTE_CARLO

    encoder = GaussianEncoder(encoder_net)
    
    # Choose decoder based on config
    if hasattr(config, 'decoder_type') and config.decoder_type == DecoderType.GAUSSIAN:
        # For Part B: Gaussian decoder for standard MNIST (continuous values)
        # Need to remove Unflatten layer for Gaussian decoder (expects flat output)
        decoder_net_gaussian = nn.Sequential(
            nn.Linear(M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
        )
        decoder = GaussianDecoder(decoder_net_gaussian, fixed_sigma=0.1)
    else:
        # For Part A: Bernoulli decoder for binary MNIST
        decoder = BernoulliDecoder(decoder_net)
    
    beta = config.beta if hasattr(config, 'beta') else 1.0
    return VAE(prior, decoder, encoder, kl_mode, beta=beta)


def _build_flow_prior(M: int, num_layers: int, num_hidden: int) -> FlowPrior:
    base = GaussianBase(M)
    mask = torch.zeros(M)
    mask[M // 2 :] = 1
    transformations = []
    for _ in range(num_layers):
        mask = 1 - mask
        scale_net = nn.Sequential(
            nn.Linear(M, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, M),
            nn.Tanh(),
        )
        translation_net = nn.Sequential(
            nn.Linear(M, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, M),
        )
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
    return FlowPrior(Flow(base, transformations))


def _build_ddpm(config: DDPMConfig) -> nn.Module:
    from src.unet import Unet
    from src.ddpm import DDPM

    network = Unet()
    return DDPM(network, beta_1=config.beta_1, beta_T=config.beta_T, T=config.T)

def _build_latent_ddpm(config: LatentDDPMConfig) -> nn.Module:
    from src.ddpm import DDPM, FcNetwork

    # For latent DDPM, use FcNetwork that operates on latent vectors
    # instead of UNet which expects 28x28 images
    network = FcNetwork(input_dim=config.M, num_hidden=config.num_hidden)
    return DDPM(network, beta_1=config.beta_1, beta_T=config.beta_T, T=config.T)