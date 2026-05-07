from collections import Counter

import networkx as nx
from torch_geometric.data import Data

from utils.logger import get_logger

logger = get_logger(__name__)


def pyg_to_nx(data: Data) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist()
    G.add_edges_from((u, v) for u, v in edges if u < v)
    return G


def _wl_hash(G: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(G, iterations=3)


def compute_novelty_uniqueness(
    generated: list[nx.Graph], training: list[nx.Graph]
) -> dict[str, float]:
    train_hashes = {_wl_hash(G) for G in training}
    gen_hashes = [_wl_hash(G) for G in generated]
    gen_counts = Counter(gen_hashes)

    n = len(gen_hashes)
    novel = sum(1 for h in gen_hashes if h not in train_hashes)
    unique = sum(1 for h in gen_hashes if gen_counts[h] == 1)
    novel_and_unique = sum(
        1 for h in gen_hashes if h not in train_hashes and gen_counts[h] == 1
    )

    results = {
        "novel": novel / n * 100,
        "unique": unique / n * 100,
        "novel_and_unique": novel_and_unique / n * 100,
    }
    logger.info(
        "Novelty/uniqueness",
        novel=f"{results['novel']:.1f}%",
        unique=f"{results['unique']:.1f}%",
        novel_and_unique=f"{results['novel_and_unique']:.1f}%",
    )
    return results


def _eigenvector_centrality(G: nx.Graph) -> dict[int, float]:
    try:
        return nx.eigenvector_centrality(G, max_iter=1000)
    except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
        return {n: 0.0 for n in G.nodes()}


def compute_graph_statistics(graphs: list[nx.Graph]) -> dict[str, list[float]]:
    degrees = []
    clustering = []
    centrality = []

    for G in graphs:
        degrees.extend(d for _, d in G.degree())
        clustering.extend(nx.clustering(G).values())
        centrality.extend(_eigenvector_centrality(G).values())

    logger.info(
        "Graph statistics",
        num_graphs=len(graphs),
        total_nodes=len(degrees),
    )
    return {
        "node_degree": degrees,
        "clustering_coefficient": clustering,
        "eigenvector_centrality": centrality,
    }
