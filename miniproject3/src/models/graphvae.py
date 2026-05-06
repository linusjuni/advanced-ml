import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers=2):
        super().__init__()
        convs = []
        for i in range(num_layers):
            convs.append(GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim))
        self.convs = nn.ModuleList(convs)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * max_nodes),
        )

    def forward(self, z):
        B   = z.size(0)
        adj = self.fc(z).view(B, self.max_nodes, self.max_nodes)
        adj = (adj + adj.transpose(1, 2)) / 2
        return adj


class GraphVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, max_nodes):
        super().__init__()
        self.encoder    = Encoder(in_dim, hidden_dim, latent_dim)
        self.decoder    = Decoder(latent_dim, hidden_dim, max_nodes)
        self.max_nodes  = max_nodes
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x, edge_index, batch):
        mu, logvar    = self.encoder(x, edge_index, batch)
        A             = to_dense_adj(edge_index, batch, max_num_nodes=self.max_nodes)
        X_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
        z             = self.reparameterize(mu, logvar)
        adj_logits    = self.decoder(z)
        return adj_logits, A, mu, logvar, mask

    def loss(self, adj_logits, A, mu, logvar, mask):
        node_mask  = mask.unsqueeze(1) & mask.unsqueeze(2)

        recon_loss = F.binary_cross_entropy_with_logits(
            adj_logits, A, reduction="none"
        )
        recon_loss = (recon_loss * node_mask.float()).sum(dim=(1, 2)).mean()

        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()

        return recon_loss + kl, recon_loss, kl

    @torch.no_grad()
    def sample(self, num_samples: int, device, node_counts: np.ndarray, probs: np.ndarray) -> list[nx.Graph]:
        """Sample graphs with variable node counts drawn from the empirical training distribution.

        For each sample: draw N ~ Categorical(node_counts, probs), decode a latent z,
        threshold the top-left N×N submatrix of the adjacency logits, and return as nx.Graph.
        """
        rng    = np.random.default_rng()
        graphs = []
        for _ in range(num_samples):
            n          = int(rng.choice(node_counts, p=probs))
            z          = torch.randn(1, self.latent_dim).to(device)
            adj_logits = self.decoder(z)[0]          # (max_nodes, max_nodes)
            adj        = (torch.sigmoid(adj_logits[:n, :n]) > 0.5).float()
            adj        = torch.triu(adj, diagonal=1)
            adj        = (adj + adj.T).cpu().numpy()
            graphs.append(nx.from_numpy_array(adj))
        return graphs
