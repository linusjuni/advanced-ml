# Part A Progress

## Code
- VAE architectures: Gaussian, MoG (K=10), Flow (5 layers, 64 hidden) priors — all complete
- Training pipeline (`train_vae.py`) — complete
- `evaluate_vae.py` — loads all checkpoints per prior, computes test ELBO (mean ± std), generates sample grids and posterior plots for seed-1 model; results saved to `src/results/part_a/`
- `utils/viz_utils.py` — seaborn whitegrid/muted theme; `plot_training_curves`, `plot_sample_grid`, `project_to_2d` (PCA helper), `plot_prior_and_aggregate_posterior` (overlaid prior + posterior, PCA for M>2, subsampled, coloured by digit class)
- `evaluate_ddpm.py` imports fixed to use `src.` prefix

## Trained models
- 10 models each of Gaussian, MoG, and Flow priors (M=32, seeds 1-10)
- Config: epochs=20, bs=128, lr=1e-3, beta=1.0, bernoulli decoder, binarized MNIST

## Results (Part A)
- Test ELBO (mean ± std over 10 seeds):
  - Gaussian: -82.80 ± 0.41
  - MoG:      -81.63 ± 0.66
  - Flow:     -79.88 ± 0.33
- Flow > MoG > Gaussian — more flexible prior → better ELBO and more structured latent space
- Sample grids and posterior plots saved to `src/results/part_a/`

## Still TODO
- **Report**: figures, tables, code snippets, discussion
