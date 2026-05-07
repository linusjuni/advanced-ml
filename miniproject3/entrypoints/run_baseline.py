import json
from datetime import datetime
from pathlib import Path

from src.data import load_mutag
from src.baseline import build_node_count_distribution, compute_link_probability
from utils.logger import get_logger
from utils.settings import settings

logger = get_logger(__name__)


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(settings.OUTPUT_DIR) / "baseline" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    train_data, _ = load_mutag()

    node_counts, probs = build_node_count_distribution(train_data)
    logger.info(
        "Node-count distribution",
        num_distinct=len(node_counts),
        range=f"[{node_counts[0]}, {node_counts[-1]}]",
    )

    link_probs = {}
    for n in node_counts:
        r = compute_link_probability(train_data, int(n))
        count = sum(1 for d in train_data if d.num_nodes == n)
        link_probs[int(n)] = r
        logger.info(f"  N={n:3d}", r=f"{r:.4f}", graphs=count)

    baseline_params = {
        "node_counts": node_counts.tolist(),
        "probs": probs.tolist(),
        "link_probabilities": {str(k): v for k, v in link_probs.items()},
        "num_train_graphs": len(train_data),
    }
    params_path = run_dir / "baseline_params.json"
    with open(params_path, "w") as f:
        json.dump(baseline_params, f, indent=2)
    logger.success("Saved baseline parameters", path=str(params_path))


if __name__ == "__main__":
    main()
