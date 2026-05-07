# Visualisations

## 1. Novelty & uniqueness table
Compares baseline (Erdős-Rényi) and GraphVAE on three metrics across 1000 sampled graphs: novel (not in training set), unique (no duplicates), and novel+unique combined.

## 2. Graph statistics histogram grid (3×3)
Histograms of node degree, clustering coefficient, and eigenvector centrality for three distributions side by side: empirical (training data), baseline, and GraphVAE. All metrics use the same bins per row for easy visual comparison.

Output: `outputs/evaluation/<timestamp>/histogram_grid.pdf`

## 3. Qualitative sample grid (2×4)
Four real MUTAG graphs (top row, increasing node count) next to four GraphVAE-generated graphs (bottom row). Shows structural differences at a glance — real graphs are sparse and chain-like; generated graphs tend to be denser.

Output: `outputs/train_gvae/<timestamp>/sample_grid.pdf`
