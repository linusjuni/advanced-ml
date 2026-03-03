import torch
import torch.nn as nn
class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also takes time as an argument.

        Parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden),
            nn.LayerNorm(num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LayerNorm(num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, input_dim),
        )

    def forward(self, x, t):
        """
        Forward function for the network.

        Parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)