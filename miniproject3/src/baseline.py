import numpy as np
import networkx as nx
from collections import Counter
from torch_geometric.data import Data

from utils.logger import get_logger

logger = get_logger(__name__)


def build_node_count_distribution(train_data: list[Data]) -> tuple[np.ndarray, np.ndarray]:
    """Baseline step 1: build the empirical distribution over node counts in training graphs.

    Returns parallel arrays (node_counts, probs) so that probs[i] is the fraction
    of training graphs that have exactly node_counts[i] nodes.
    """
    counter = Counter(d.num_nodes for d in train_data)
    node_counts = np.array(sorted(counter.keys()))
    probs = np.array([counter[n] for n in node_counts], dtype=float)
    probs /= probs.sum()
    return node_counts, probs


def compute_link_probability(train_data: list[Data], n: int) -> float:
    """Baseline step 2: compute graph density r for training graphs with exactly N nodes.

    Pools all N-node training graphs and computes:
        r = total_edges / total_possible_edges
    where total_possible = sum of N(N-1)/2 per graph. The //2 on edge_index corrects
    for PyG storing each undirected edge in both directions.
    """
    graphs_with_n = [d for d in train_data if d.num_nodes == n]
    total_edges = sum(d.edge_index.shape[1] // 2 for d in graphs_with_n)
    total_possible = sum(d.num_nodes * (d.num_nodes - 1) // 2 for d in graphs_with_n)
    return total_edges / total_possible


class ErdosRenyiBaseline:
    """Erdős-Rényi baseline combining all three steps from project section 2.2.

    On each call to sample():
      1. Draw N from the empirical node-count distribution of the training data.
      2. Look up r for that N (graph density of N-node training graphs).
      3. Return nx.erdos_renyi_graph(N, r): each of N(N-1)/2 edges included independently with prob r.
    """

    def __init__(self, train_data: list[Data]):
        self.train_data = train_data
        self.node_counts, self.probs = build_node_count_distribution(train_data)
        self._rng = np.random.default_rng()
        logger.info(
            f"ErdosRenyiBaseline ready: {len(self.node_counts)} distinct node counts "
            f"in [{self.node_counts[0]}, {self.node_counts[-1]}]."
        )

    def sample(self) -> nx.Graph:
        n = int(self._rng.choice(self.node_counts, p=self.probs))
        r = compute_link_probability(self.train_data, n)
        return nx.erdos_renyi_graph(n=n, p=r)

    def sample_many(self, num_samples: int) -> list[nx.Graph]:
        graphs = [self.sample() for _ in range(num_samples)]
        logger.success(f"Sampled {num_samples} graphs from Erdős-Rényi baseline.")
        return graphs
