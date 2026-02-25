from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

from src.vae import VAE, GaussianEncoder, BernoulliDecoder, KLMode
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


@dataclass(kw_only=True)
class VAEBaseConfig:
    M: int
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
    # TODO (Part B)
    pass


@dataclass(kw_only=True)
class LatentDDPMConfig(DDPMBaseConfig):
    # TODO (Part B)
    model_type: ModelType = ModelType.LATENT_DDPM  # override model_type


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
    if isinstance(config, VAEBaseConfig):
        parts.append(config.prior.value)
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
    decoder = BernoulliDecoder(decoder_net)
    return VAE(prior, decoder, encoder, kl_mode)


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
    # TODO (Part B)
    raise NotImplementedError("DDPM loading not yet implemented")


def _build_latent_ddpm(config: LatentDDPMConfig) -> nn.Module:
    # TODO (Part B)
    raise NotImplementedError("Latent DDPM loading not yet implemented")
