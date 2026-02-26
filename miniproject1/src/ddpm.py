# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(
            self.alpha.cumprod(dim=0), requires_grad=False
        )

    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """
        # t ~ Uniform({1, ..., T})
        t = torch.randint(1, self.T + 1, (x.shape[0],), device=x.device)

        # ε ~ N(0, I)
        epsilon = torch.randn_like(x)

        # Grab ᾱ_t for each sample, shape (batch_size, 1)
        alpha_cumprod_t = self.alpha_cumprod[t - 1].unsqueeze(-1)

        # Build x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        x_t = (
            torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * epsilon
        )

        # Predict the noise, normalize t to [0, 1]
        epsilon_pred = self.network(x_t, t.unsqueeze(-1).float() / self.T)

        # Return ||ε - ε_θ||² per sample
        neg_elbo = (epsilon - epsilon_pred).pow(2).sum(dim=-1)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in reversed(range(self.T)):
            # z ~ N(0, I) if t > 0, else z = 0
            z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            # Predict the noise
            t_tensor = torch.full(
                (x_t.shape[0], 1), (t + 1) / self.T, device=x_t.device
            )
            epsilon_pred = self.network(x_t, t_tensor)

            # x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ) + σ_t * z
            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            sigma_t = torch.sqrt(self.beta[t])

            # Compute x_{t-1}
            x_t = (1 / torch.sqrt(alpha_t)) * (
                x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * epsilon_pred
            ) + sigma_t * z

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


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
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
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


def train(model, optimizer, data_loader, epochs, device) -> list[float]:
    """
    Train a DDPM model.

    Parameters:
    model: [DDPM]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.

    Returns:
    epoch_losses: [list[float]]
        Average training loss per epoch.
    """
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for x in data_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():12.4f}", epoch=f"{epoch + 1}/{epochs}"
            )
            progress_bar.update()

        epoch_losses.append(epoch_loss / num_batches)

    return epoch_losses