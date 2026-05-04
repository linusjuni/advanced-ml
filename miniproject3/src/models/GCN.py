import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, filter_length=3, dropout=0.3):
        super().__init__()

        self.filter_length = filter_length

        self.h = nn.Parameter(1e-5 * torch.randn(filter_length))
        self.h.data[0] = 1.

        self.mlp_in = MLP(in_dim, hidden_dim, hidden_dim, dropout)
        self.mlp_out = MLP(hidden_dim, hidden_dim, out_dim, dropout)

        self.residual = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim else nn.Identity()
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, X, A):
        H = self.mlp_in(X)

        Ak = H
        out = self.h[0] * Ak

        for k in range(1, self.filter_length):
            Ak = A @ Ak
            out = out + self.h[k] * Ak

        out = self.mlp_out(out)

        return self.norm(out + self.residual(X))

class Encoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, latent_dim):
        super().__init__()

        self.conv1 = GraphConvLayer(node_feature_dim, hidden_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, X, A, mask=None):
        H = self.conv1(X, A)
        H = F.relu(H)

        H = self.conv2(H, A)
        H = F.relu(H)

        # graph-level pooling
        if mask is not None:
            H = H * mask.unsqueeze(-1).float()
        g = H.sum(dim=1)

        mu = self.fc_mu(g)
        logvar = self.fc_logvar(g)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_nodes):
        super().__init__()

        self.max_nodes = max_nodes

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes)
        )

    def forward(self, z):
        B = z.size(0)

        edge_logits = self.fc(z).view(B, self.max_nodes, self.max_nodes)

        return edge_logits


class GraphVAE(nn.Module):
    """
    Encoder:
        q_phi(z|G) = N(mu(G), diag(sigma^2(G)))

    Prior:
        p(z) = N(0, I)

    Decoder:
        p_theta(G|z):
            A_uv ~ Bernoulli(sigmoid(score_uv(z)))

    ELBO:
        E_q[ log p_theta(G|z) ] - KL(q(z|G)||p(z))
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        max_nodes=None,
        latent_dim=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim

    # reparameterization trick
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, edge_index, batch):
        if self.max_nodes is None:
            raise ValueError("GraphVAE.max_nodes must be set before calling forward().")
        A = to_dense_adj(
            edge_index,
            batch,
            max_num_nodes=self.max_nodes
        )

        X, mask = to_dense_batch(
            x,
            batch,
            max_num_nodes=self.max_nodes
        )

        mu, logvar = self.encoder(X, A, mask)

        z = self.reparameterize(mu, logvar)

        adj_logits = self.decoder(z)

        return adj_logits, A, mu, logvar, mask

    # ------------------------------------------------------
    # ELBO LOSS
    # L = E_q [ log p_theta(G|z) ] - KL[q(z|G)||p(z)]
    # ------------------------------------------------------
    def loss(self, adj_logits, A_true, mu, logvar, mask):
        # only valid nodes
        node_mask = mask.unsqueeze(1) & mask.unsqueeze(2)

        # reconstruction term:
        # log p(G|z) using Bernoulli likelihood
        recon = F.binary_cross_entropy_with_logits(
            adj_logits,
            A_true,
            reduction="none"
        )

        recon = recon * node_mask.float()
        recon = recon.sum(dim=(1, 2)).mean()

        # KL[q(z|G)||p(z)]
        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        ).mean()

        # negative ELBO to minimize
        loss = recon + kl

        return loss, recon, kl

    @torch.no_grad()
    def sample(self, num_samples, device):

        z = torch.randn(num_samples, self.latent_dim).to(device)

        adj_logits = self.decoder(z)

        probs = torch.sigmoid(adj_logits)

        A = (probs > 0.5).float()

        # symmetrize
        A = torch.triu(A, diagonal=1)
        A = A + A.transpose(1, 2)

        return A

