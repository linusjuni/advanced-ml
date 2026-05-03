import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data

from utils.logger import get_logger
from utils.settings import settings

logger = get_logger(__name__)


def load_mutag(train_ratio: float = 0.8) -> tuple[list[Data], list[Data]]:
    """Load MUTAG and return a seeded 80/20 train/test split.

    Each Data object contains:
      - edge_index [2, 2E]: directed COO pairs (each undirected edge stored twice)
      - x         [N, 7]:  one-hot atom type (C, N, O, F, I, Cl, Br)
      - y         [1]:     binary mutagenicity label
    """
    dataset = TUDataset(root=str(settings.DATA_DIR), name="MUTAG")
    rng = np.random.default_rng(settings.RANDOM_SEED)
    indices = rng.permutation(len(dataset))
    n_train = int(len(dataset) * train_ratio)
    train_data = [dataset[int(i)] for i in indices[:n_train]]
    test_data = [dataset[int(i)] for i in indices[n_train:]]
    logger.info(f"Loaded MUTAG: {len(train_data)} train, {len(test_data)} test graphs.")
    return train_data, test_data
