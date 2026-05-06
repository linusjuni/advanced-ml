from src.data import load_mutag
from src.baseline import ErdosRenyiBaseline, build_node_count_distribution, compute_link_probability

train_data, test_data = load_mutag()

# Step 1: empirical node-count distribution
node_counts, probs = build_node_count_distribution(train_data)
print(f"\nStep 1 — empirical node-count distribution ({len(train_data)} training graphs):")
for n, p in zip(node_counts, probs):
    print(f"  N={n:3d}: {p:.3f} {'#' * int(p * 50)}")

# Step 2: link probability r for each observed N
print("\nStep 2 — link probability r per N (total edges / total possible edges):")
for n in node_counts:
    r = compute_link_probability(train_data, n)
    count = sum(1 for d in train_data if d.num_nodes == n)
    print(f"  N={n:3d}: r={r:.4f}  ({count} graph{'s' if count > 1 else ''})")

# Step 3: sample one graph end-to-end
baseline = ErdosRenyiBaseline(train_data)
g = baseline.sample()
print(f"\nStep 3 — sampled Erdős-Rényi graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
