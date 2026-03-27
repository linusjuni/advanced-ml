# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

        Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters:
        decoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M)` as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.mean(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)

    def mean(self, z):
        """Return decoder mean f(z) for pull-back and ensemble geometry computations."""
        return self.decoder_net(z)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x)
        z = q.rsample()
        return torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def new_encoder(M):
    """
    Construct a convolutional encoder network for 28x28 grayscale images.

    Parameters:
    M: [int]
       Dimension of the latent space. Output dimension is 2*M (mean and log-std).
    """
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2 * M),
    )


def new_decoder(M):
    """
    Construct a transposed-convolutional decoder network producing 28x28 grayscale images.

    Parameters:
    M: [int]
       Dimension of the latent space.
    """
    return nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )


class EnsembleGaussianDecoder(nn.Module):
    def __init__(self, decoder_nets):
        """
        Define an ensemble Gaussian decoder with multiple decoder networks.

        Parameters:
        decoder_nets: [list[torch.nn.Module]]
            Decoder networks that map latent variables to output means.
        """
        super().__init__()
        self.decoder_nets = nn.ModuleList(decoder_nets)
        if len(self.decoder_nets) == 0:
            raise ValueError("decoder_nets must contain at least one decoder")

    @property
    def num_decoders(self):
        return len(self.decoder_nets)

    def sample_member_idx(self, device=None):
        idx = torch.randint(0, self.num_decoders, (1,), device=device)
        return int(idx.item())

    def distribution(self, z, member_idx):
        means = self.mean(z, member_idx)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)

    def mean(self, z, member_idx):
        """Return decoder mean f_member(z) for a specific ensemble member index."""
        return self.decoder_nets[member_idx](z)

    def forward(self, z, member_idx=None):
        """
        Return p(x|z) for one decoder member.

        If member_idx is None, sample one member uniformly.
        """
        if member_idx is None:
            member_idx = self.sample_member_idx(z.device)
        return self.distribution(z, member_idx)


class VAEEnsemble(nn.Module):
    """Variational Autoencoder with one encoder and an ensemble of decoders."""

    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x, member_idx=None):
        q = self.encoder(x)
        z = q.rsample()
        px = self.decoder(z, member_idx=member_idx)
        return torch.mean(px.log_prob(x) - q.log_prob(z) + self.prior().log_prob(z))

    def sample(self, n_samples=1, member_idx=None):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z, member_idx=member_idx).sample()

    def forward(self, x, member_idx=None):
        return -self.elbo(x, member_idx=member_idx)


def new_ensemble_decoder(M, num_decoders):
    """Build an ensemble of decoder networks."""
    return EnsembleGaussianDecoder([new_decoder(M) for _ in range(num_decoders)])


def build_ensemble_vae(latent_dim, num_decoders):
    """Convenience builder for a VAE ensemble with Gaussian prior, Gaussian encoder, and ensemble of Gaussian decoders."""
    return VAEEnsemble(
        GaussianPrior(latent_dim),
        new_ensemble_decoder(latent_dim, num_decoders),
        GaussianEncoder(new_encoder(latent_dim)),
    )
