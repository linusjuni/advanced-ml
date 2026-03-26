import torch


def compute_geodesic(x_start, x_end, decoder, n_points=10, n_steps=200):
    """
    Compute a geodesic curve between two latent points under the pull-back metric
    induced by the decoder mean. The geodesic is found by minimizing the discrete
    curve energy E = sum_i ||f(z_{i+1}) - f(z_i)||^2, where f is the decoder mean.

    Parameters:
    x_start: [torch.Tensor]
        Starting point in latent space, shape (M,).
    x_end: [torch.Tensor]
        Ending point in latent space, shape (M,).
    decoder: [torch.nn.Module]
        The decoder module whose mean defines the pull-back metric.
    n_points: [int]
        Number of points along the discretized curve (including endpoints).
    n_steps: [int]
        Number of LBFGS optimization steps.

    Returns:
    full_curve: [torch.Tensor]
        Optimized curve in latent space, shape (n_points, M).
    """
    t = torch.linspace(0, 1, n_points, device=x_start.device)

    # Linear interpolation as initial guess
    curve = torch.stack([(1 - ti) * x_start + ti * x_end for ti in t])

    # Only optimize interior points; keep endpoints fixed
    interior = curve[1:-1].clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([interior], lr=1)

    def energy_loss():
        optimizer.zero_grad()
        full_curve = torch.cat([
            x_start.unsqueeze(0),
            interior,
            x_end.unsqueeze(0),
        ])
        decoded = decoder(full_curve).mean
        diff = decoded[1:] - decoded[:-1]
        energy = (diff ** 2).sum()
        energy.backward()
        return energy

    for _ in range(n_steps):
        optimizer.step(energy_loss)

    with torch.no_grad():
        full_curve = torch.cat([
            x_start.unsqueeze(0),
            interior,
            x_end.unsqueeze(0),
        ])
    return full_curve


def curve_length(curve, decoder):
    """
    Compute the geodesic length of a discrete latent curve as the sum of
    decoded segment norms: L = sum_i ||f(z_{i+1}) - f(z_i)||.

    Parameters:
    curve: [torch.Tensor]
        Discrete curve in latent space, shape (n_points, M).
    decoder: [torch.nn.Module]
        The decoder module whose mean defines the pull-back metric.

    Returns:
    length: [float]
    """
    with torch.no_grad():
        decoded = decoder(curve).mean
        diff = decoded[1:] - decoded[:-1]
        return diff.flatten(1).norm(dim=1).sum().item()
