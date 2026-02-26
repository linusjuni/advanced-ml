# Priors for VAE models
# GaussianPrior from DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024

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
        super(GaussianPrior, self).__init__()
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

    def log_prob(self, z):
        return self.forward().log_prob(z)

    def sample(self, sample_shape=torch.Size()):
        return self.forward().sample(sample_shape)


class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussians prior distribution.

        Parameters:
        M: [int]
           Dimension of the latent space.
        K: [int]
           Number of mixture components.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K
        self.logits = nn.Parameter(torch.zeros(K))
        self.means = nn.Parameter(torch.randn(K, M))
        self.log_stds = nn.Parameter(torch.zeros(K, M))

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(loc=self.means, scale=torch.exp(self.log_stds)), 1)
        return td.MixtureSameFamily(mix, comp)

    def log_prob(self, z):
        return self.forward().log_prob(z)

    def sample(self, sample_shape=torch.Size()):
        return self.forward().sample(sample_shape)


class FlowPrior(nn.Module):
    def __init__(self, flow):
        """
        A normalizing flow used as a prior distribution.

        Parameters:
        flow: [Flow]
            A normalizing flow model with log_prob(z) and sample(shape) methods.
        """
        super(FlowPrior, self).__init__()
        self.flow = flow

    def log_prob(self, z):
        return self.flow.log_prob(z)

    def sample(self, sample_shape=torch.Size()):
        return self.flow.sample(sample_shape)
