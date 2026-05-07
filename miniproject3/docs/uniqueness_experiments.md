# Uniqueness Experiments

## Problem
After training, sampling 1000 graphs from GraphVAE gave only **22.3% unique** graphs. The model kept generating near-identical structures.

## What didn't work

All three attempts made things worse by adding more KL regularisation pressure:

| Config | Uniqueness |
|---|---|
| Shorter KL warmup (5 epochs) + weight decay 0.01 | 0.9% |
| No warmup + smaller model (64/32 dims) | 8.4% |
| Higher beta (β=2.0) | 9.0% |

The root cause wasn't the training — KL was not collapsed. More regularisation just hurt reconstruction without fixing diversity.

## What worked

The real problem was in **sampling**, not training. The decoder used a hard threshold (`sigmoid(logit) > 0.5`), making decoding deterministic: the same or similar `z` always produced the same binary adjacency matrix.

**Fix:** Replace the hard threshold with Bernoulli edge sampling — sample each edge independently from `Bernoulli(sigmoid(logit))`. This is also the principled approach consistent with how the model was trained.

**Result: 22.3% → 100% unique. No retraining needed.**
