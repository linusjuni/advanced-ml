# %% Programming excercise: Shallow embedding — Competition version

# %% Import libraries
import torch
import itertools
from tqdm import tqdm

# %% Device
device = 'cpu'  # SVD needs CPU; small graph so no speedup from MPS

# %% Load graph data
A = torch.load('data.pt', weights_only=True).float()
n_nodes = A.shape[0]
n_pairs = n_nodes * (n_nodes - 1) // 2
idx_all_pairs = torch.triu_indices(n_nodes, n_nodes, 1)
target = A[idx_all_pairs[0], idx_all_pairs[1]].float()

# %% SVD initialization helper
def svd_init(A, d):
    """Compute rank-d SVD of adjacency matrix, return scaled left singular vectors."""
    U, S, _ = torch.linalg.svd(A)
    # X = U[:, :d] * sqrt(S[:d]) so that X @ X^T ≈ A
    return U[:, :d] * S[:d].sqrt().unsqueeze(0)

# %% Shallow node embedding with SVD init support
class Shallow(torch.nn.Module):
    def __init__(self, n_nodes, embedding_dim, init_embeddings=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_nodes, embedding_dim=embedding_dim)
        if init_embeddings is not None:
            self.embedding.weight.data = init_embeddings.clone()
        self.bias = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, rx, tx):
        return torch.sigmoid(self.logits(rx, tx))

    def logits(self, rx, tx):
        return (self.embedding.weight[rx] * self.embedding.weight[tx]).sum(1) + self.bias

# %% Training function
def train_model(n_nodes, embedding_dim, train_rx, train_tx, train_target,
                init_embeddings=None, lr=1e-2, weight_decay=0.0, max_step=5000, verbose=False):
    model = Shallow(n_nodes, embedding_dim, init_embeddings=init_embeddings)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = torch.nn.BCELoss()

    for i in range(max_step):
        prob = model(train_rx, train_tx)
        loss = bce(prob, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if verbose:
        print(f'  Final train loss: {loss.item():.4f}')
    return model

# %% Evaluate on validation set
def eval_model(model, val_rx, val_tx, val_target):
    bce = torch.nn.BCELoss()
    with torch.no_grad():
        prob = model(val_rx, val_tx)
        return bce(prob, val_target).item()

# ============================================================
# STEP 1: K-fold CV grid search
# ============================================================
print("=" * 60)
print("STEP 1: K-fold CV grid search")
print("=" * 60)

param_grid = {
    'embedding_dim': [2, 3, 4, 5, 6, 8, 10],
    'weight_decay': [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    'lr': [1e-2],
    'max_step': [5000],
}

K = 5
torch.manual_seed(0)
perm = torch.randperm(n_pairs)
fold_size = n_pairs // K

best_val_loss = float('inf')
best_params = None

keys = list(param_grid.keys())
combos = list(itertools.product(*[param_grid[k] for k in keys]))
print(f"Testing {len(combos)} hyperparameter combinations x {K} folds")

for combo in tqdm(combos, desc="Grid search"):
    params = dict(zip(keys, combo))
    fold_losses = []

    for fold in range(K):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_mask = perm[val_start:val_end]
        train_mask = torch.cat([perm[:val_start], perm[val_end:]])

        trx, ttx = idx_all_pairs[0][train_mask], idx_all_pairs[1][train_mask]
        tt = target[train_mask]
        vrx, vtx = idx_all_pairs[0][val_mask], idx_all_pairs[1][val_mask]
        vt = target[val_mask]

        init_emb = svd_init(A, params['embedding_dim'])
        model = train_model(
            n_nodes, params['embedding_dim'], trx, ttx, tt,
            init_embeddings=init_emb,
            lr=params['lr'], weight_decay=params['weight_decay'],
            max_step=params['max_step']
        )
        fold_losses.append(eval_model(model, vrx, vtx, vt))

    mean_loss = sum(fold_losses) / len(fold_losses)
    if mean_loss < best_val_loss:
        best_val_loss = mean_loss
        best_params = params

print(f"\nBest params: {best_params}")
print(f"Best CV loss: {best_val_loss:.4f}")

# ============================================================
# STEP 2: Ensemble training on ALL data with best hyperparams
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Ensemble training on full data")
print("=" * 60)

N_ENSEMBLE = 20
all_logits = []

rx_all, tx_all = idx_all_pairs[0], idx_all_pairs[1]

for i in tqdm(range(N_ENSEMBLE), desc="Ensemble"):
    torch.manual_seed(i * 137)  # different seed each model
    init_emb = svd_init(A, best_params['embedding_dim'])
    # Add small noise to SVD init so each model is slightly different
    init_emb = init_emb + 0.01 * torch.randn_like(init_emb)

    model = train_model(
        n_nodes, best_params['embedding_dim'],
        rx_all, tx_all, target,
        init_embeddings=init_emb,
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        max_step=best_params['max_step'],
        verbose=(i == 0)
    )
    with torch.no_grad():
        all_logits.append(model.logits(rx_all, tx_all))

# Average in logit space
avg_logits = torch.stack(all_logits).mean(dim=0)

# ============================================================
# STEP 3: Temperature scaling
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Temperature scaling")
print("=" * 60)

# Use a fresh 80/20 split for temperature calibration
torch.manual_seed(99)
perm_temp = torch.randperm(n_pairs)
n_cal_train = int(0.8 * n_pairs)
cal_train_idx = perm_temp[:n_cal_train]
cal_val_idx = perm_temp[n_cal_train:]

# Train a fresh ensemble on cal_train, calibrate T on cal_val
cal_logits_list = []
cal_rx, cal_tx = idx_all_pairs[0][cal_train_idx], idx_all_pairs[1][cal_train_idx]
cal_target_train = target[cal_train_idx]

for i in tqdm(range(N_ENSEMBLE), desc="Calibration ensemble"):
    torch.manual_seed(i * 137 + 1000)
    init_emb = svd_init(A, best_params['embedding_dim'])
    init_emb = init_emb + 0.01 * torch.randn_like(init_emb)

    model = train_model(
        n_nodes, best_params['embedding_dim'],
        cal_rx, cal_tx, cal_target_train,
        init_embeddings=init_emb,
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
        max_step=best_params['max_step'],
    )
    with torch.no_grad():
        # Get logits on ALL pairs (we'll index into val portion)
        cal_logits_list.append(model.logits(idx_all_pairs[0], idx_all_pairs[1]))

cal_avg_logits = torch.stack(cal_logits_list).mean(dim=0)
cal_val_logits = cal_avg_logits[cal_val_idx]
cal_val_target = target[cal_val_idx]

# Optimize temperature T
temperature = torch.nn.Parameter(torch.tensor([1.0]))
temp_optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=500)
bce = torch.nn.BCELoss()

def temp_closure():
    temp_optimizer.zero_grad()
    scaled = torch.sigmoid(cal_val_logits / temperature)
    loss = bce(scaled, cal_val_target)
    loss.backward()
    return loss

temp_optimizer.step(temp_closure)

T = temperature.item()
print(f"Learned temperature: {T:.4f}")

# Apply temperature to the full-data ensemble logits
final_probs = torch.sigmoid(avg_logits / T)

# ============================================================
# STEP 4: Save predictions
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Save predictions")
print("=" * 60)

# Sanity check
with torch.no_grad():
    final_loss = bce(final_probs, target)
    print(f"Final BCE on all data: {final_loss.item():.4f}")

torch.save(final_probs.detach(), 'link_probability.pt')
print("Saved link_probability.pt")
print(f"Predictions range: [{final_probs.min().item():.4f}, {final_probs.max().item():.4f}]")
print(f"Predictions mean:  {final_probs.mean().item():.4f}")
