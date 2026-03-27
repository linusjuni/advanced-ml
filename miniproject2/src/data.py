import torch
import torch.utils.data
from torchvision import datasets, transforms

from utils.logger import get_logger
from utils.settings import settings

logger = get_logger(__name__)


def subsample(data, targets, num_data, num_classes):
    """
    Subsample a dataset to a given number of classes and observations.

    Parameters:
    data: [torch.Tensor]
        Raw image data tensor of shape (N, H, W), uint8.
    targets: [torch.Tensor]
        Class labels of shape (N,).
    num_data: [int]
        Maximum number of observations to keep.
    num_classes: [int]
        Only keep observations with class label < num_classes.

    Returns:
    dataset: [torch.utils.data.TensorDataset]
    """
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
    new_targets = targets[idx][:num_data]
    logger.info(f"Subsampled dataset to {len(new_data)} observations with {num_classes} classes.")
    return torch.utils.data.TensorDataset(new_data, new_targets)


def load_mnist(batch_size, num_train_data=2048, num_classes=3):
    """
    Load and subsample MNIST, returning train and test data loaders.

    Parameters:
    batch_size: [int]
        Batch size for both loaders.
    num_train_data: [int]
        Number of training (and test) observations to keep (default: 2048).
    num_classes: [int]
        Number of classes to include, using labels 0..num_classes-1 (default: 3).

    Returns:
    train_loader: [torch.utils.data.DataLoader]
    test_loader: [torch.utils.data.DataLoader]
    """
    to_tensor = transforms.Compose([transforms.ToTensor()])
    train_tensors = datasets.MNIST(settings.DATA_DIR, train=True, download=True, transform=to_tensor)
    test_tensors = datasets.MNIST(settings.DATA_DIR, train=False, download=True, transform=to_tensor)

    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    test_data = subsample(test_tensors.data, test_tensors.targets, num_train_data, num_classes)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    logger.info(f"Loaded MNIST with {num_classes} classes.")
    return train_loader, test_loader
