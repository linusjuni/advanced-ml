# Plan for winning D.5

## Key insight
The evaluation is on ground truth probabilities, not binary links. We need well-calibrated probabilities, not just good classification.

## Strategy

1. **Cross-validated hyperparameter tuning**
   - K-fold CV over: embedding_dim, weight_decay, learning_rate, n_steps
   - From D.4 we know optimal dim is around 4-5 without regularization
   - With weight decay, higher dims may work better

2. **Weight decay (L2 regularization)**
   - Prevents embeddings from growing large and pushing sigmoid to 0/1
   - Keeps predictions calibrated toward the true probabilities
   - Enables using higher embedding dimensions without overfitting

3. **Train final model on ALL data**
   - Val split is only for hyperparameter selection
   - Final submission model uses all 19,900 pairs
   - Fair game per exercise instructions ("any other updates, hacks and modifications")

4. **Ensemble over random seeds**
   - Train N models with different initializations, average predictions
   - Reduces variance for free

5. **SVD initialization of embeddings**
   - Compute rank-d SVD of adjacency matrix A
   - Use top-d left singular vectors (scaled by sqrt of singular values) as initial embeddings
   - Under RDPG model, this is already a near-optimal estimate of latent positions
   - Massively speeds up convergence and avoids bad local minima from random init

6. **Temperature scaling (post-hoc calibration)**
   - After training, learn a single scalar T on a validation set that rescales logits: sigmoid(logit / T)
   - Corrects overconfident predictions pushed toward 0/1
   - Directly targets probability calibration — which is exactly what the evaluation measures
   - Nearly free: one parameter optimized on held-out data after training is done

## Execution order
1. Grid search (dim, weight_decay) with 5-fold CV
2. Pick best hyperparameters
3. Train ensemble of ~20 models on full data with those hyperparameters, using SVD-initialized embeddings
4. Average predictions in logit space, apply temperature scaling
5. Save as link_probability.pt
