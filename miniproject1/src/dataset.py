from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.utils.settings import settings


def get_binarized_mnist(train: bool = True) -> Dataset:
    """Binarized MNIST dataset (pixels > 0.5 become 1, else 0)."""
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
