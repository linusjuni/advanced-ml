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

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # graph-level pooling
        x = global_mean_pool(x, batch)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

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
        adj = self.fc(z).view(B, self.max_nodes, self.max_nodes)
        return adj

# Graph VAE
class GraphVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, max_nodes):
        super().__init__()

        self.encoder = Encoder(in_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, max_nodes)

        self.max_nodes = max_nodes
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, edge_index, batch):
        mu, logvar = self.encoder(x, edge_index, batch)
        A        = to_dense_adj(edge_index, batch, max_num_nodes=self.max_nodes)
        X_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)

        z          = self.reparameterize(mu, logvar)
        adj_logits = self.decoder(z)

        return adj_logits, A, mu, logvar, mask
    
    # Loss (ELBO)
    def loss(self, adj_logits, A, mu, logvar, mask):

        node_mask = mask.unsqueeze(1) & mask.unsqueeze(2)

        recon_loss = F.binary_cross_entropy_with_logits(
            adj_logits,
            A,
            reduction="none"
        )

        recon_loss = recon_loss * node_mask.float()
        recon_loss = recon_loss.sum(dim=(1, 2)).mean()

        kl = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon_loss + kl, recon_loss, kl
    
    # Sampling
    @torch.no_grad()
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)

        adj_logits = self.decoder(z)
        probs = torch.sigmoid(adj_logits)

        A = (probs > 0.5).float()

        # symmetrize adjacency
        A = torch.triu(A, diagonal=1)
        A = A + A.transpose(1, 2)

        return A
    
if __name__ == "__main__":
    in_dim     = 10
    hidden_dim = 32
    latent_dim = 16
    max_nodes  = 20

    model = GraphVAE(in_dim, hidden_dim, latent_dim, max_nodes)

    x          = torch.randn(4, in_dim)         
    edge_index = torch.tensor([[0, 1, 2, 3],
                                [1, 0, 3, 2]])       
    batch      = torch.tensor([0, 0, 1, 1])

    adj_logits, A, mu, logvar, mask = model(x, edge_index, batch)

    print("adj_logits:", adj_logits.shape)
    print("A:         ", A.shape)            
    print("mu:        ", mu.shape)           
    print("logvar:    ", logvar.shape)       
    print("mask:      ", mask.shape)         