# Report writing guide — Mini-project 1

One content page total (Parts A + B share it). Ultra-tight on space.

---

## Shared base (write once, reference in both parts)

**Encoder/decoder MLP** — course-provided architecture, used by all VAEs in Part A *and* the β-VAE in Part B:
- Encoder: 784 → Linear(512) → ReLU → Linear(512) → ReLU → Linear(2M); outputs mean and log-std of q(z|x)
- Decoder: M → Linear(512) → ReLU → Linear(512) → ReLU → Linear(784); Bernoulli likelihood for Part A (binarized MNIST), Gaussian (σ=0.1, fixed) for Part B (standard MNIST)
- M=32 for all VAEs
- Describe once; Part B can say "same VAE architecture as Part A with Gaussian decoder"

---

## Part A

### Table (main result)
3 rows × 4 columns. Means over 10 seeds.

| Prior    | Recon          | KL             | ELBO            |
|----------|----------------|----------------|-----------------|
| Gaussian | −59.4 ± 0.8    | 23.4 ± 0.5     | −82.8 ± 0.4     |
| MoG (K=10)| −58.5 ± 1.1  | 23.2 ± 0.5     | −81.7 ± 0.7     |
| Flow     | −57.0 ± 0.6    | 22.9 ± 0.3     | −79.9 ± 0.4     |

### Figure
3-panel figure: posterior_gaussian.png, posterior_mog.png, posterior_flow.png
- Prior shown as grey KDE, aggregate posterior coloured by digit class
- Caption must mention: projected to 2 PCs (PC1 ~X%, PC2 ~Y%)

### Key sentences to write (~4 sentences total)
1. **ELBO result + decomposition insight**: "Flow > MoG > Gaussian in ELBO; the gain (Gaussian→Flow: +2.9 nats) is driven primarily by reconstruction (+2.4 nats) rather than KL reduction (−0.5 nats), suggesting the flexible prior relaxes the constraint on the encoder rather than simply closing the prior-posterior gap."
2. **Posterior plot**: "The Gaussian aggregate posterior forms a ring mismatched to the spherical prior; MoG and Flow priors adapt to this structure, reducing the prior hole problem and enabling better-separated digit clusters."

> Note: encoder/decoder architecture is identical to the week 1 exercise code — describe briefly, no need to justify design choices. Own contributions are the MoG/Flow prior implementations, MC KL estimation, and the decomposition analysis.

---

## Part B

### Figures needed
- 3×4 sample grid: 4 samples each from DDPM, latent DDPM, VAE (best from Part A = Flow prior)
- 1 plot: β-VAE prior vs latent DDPM distribution vs aggregate posterior (for discussion)

### Table needed
FID scores + sampling time:

| Model        | FID  | Samples/sec |
|--------------|------|-------------|
| VAE (Flow)   |      |             |
| DDPM         |      |             |
| Latent DDPM (β=1) |  |             |
| Latent DDPM (β=1e-6)|  |           |

Plus β ablation row(s) for latent DDPM (β = 1e-6 required by spec).

### Key sentences to write (~4 sentences total)
1. **Architecture**: "DDPM uses a UNet; latent DDPM uses a fully-connected network operating on M-dim latent codes from the β-VAE."
2. **FID + speed**: compare the three models, note VAE is fastest (single forward pass), DDPM slowest (T=1000 steps).
3. **β ablation**: discuss what happens to latent DDPM distribution at β=1e-6 vs β=1 (near-Gaussian posterior → trivial prior matching).
4. **Posterior comparison plot**: tie back to Part A — show that the latent DDPM learns a distribution that better matches the aggregate posterior than the β-VAE prior alone.

---

## Part C (code page — separate page)

Five snippets required:
1. VAE ELBO for non-Gaussian prior (MC KL) → `vae.py:elbo()`, the `else` branch
2. Flow implementation → `flow.py`: `MaskedCouplingLayer.forward/inverse` + `Flow.log_prob`
3. DDPM ELBO → `ddpm.py`
4. DDPM sampling algorithm → `ddpm.py`
5. Latent DDPM training + sampling → `train_latent_ddpm.py`

---

## Space budget (rough)
- Header (names, IDs, title): ~2 lines
- Part A table: ~5 lines
- Part A figure (3 panels): ~35% of page width as a strip
- Part A text: ~4 sentences
- Part B figures: ~35% of page
- Part B table: ~6 lines
- Part B text: ~4 sentences
- Shared architecture: ~2 sentences (can merge into Part A opening)
