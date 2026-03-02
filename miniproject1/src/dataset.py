from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.utils.settings import settings

def get_binarized_mnist(train: bool = True) -> Dataset:
    """Binarized MNIST dataset (pixels > 0.5 become 1, else 0). For Part A."""
    return datasets.MNIST(
        settings.DATA_DIR,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (0.5 < x).float().squeeze()),
            ]
        ),
    )


def get_dequantized_mnist(train: bool = True) -> Dataset:
    """Dequantized MNIST scaled to [-1, 1], flattened to 784. For Part B."""
    return datasets.MNIST(
        settings.DATA_DIR,
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 2 - 1).view(-1)),
            ]
        )
    )