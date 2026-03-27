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
        full_curve = torch.cat(
            [
                x_start.unsqueeze(0),
                interior,
                x_end.unsqueeze(0),
            ]
        )
        decoded = decoder(full_curve).mean
        diff = decoded[1:] - decoded[:-1]
        energy = (diff**2).sum()
        energy.backward()
        return energy

    for _ in range(n_steps):
        optimizer.step(energy_loss)

    with torch.no_grad():
        full_curve = torch.cat(
            [
                x_start.unsqueeze(0),
                interior,
                x_end.unsqueeze(0),
            ]
        )
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


def _sample_member_pairs(num_segments, mc_samples, num_decoders, device):
    """Sample decoder index pairs (l, k) uniformly for each segment."""
    member_l = torch.randint(0, num_decoders, (num_segments, mc_samples), device=device)
    member_k = torch.randint(0, num_decoders, (num_segments, mc_samples), device=device)
    return member_l, member_k


def _ensemble_segment_sqnorm(z_left, z_right, decoder, member_l, member_k):
    """Monte Carlo estimate of E_{l,k}[||f_l(z_left) - f_k(z_right)||^2]."""
    sq_sum = torch.zeros((), device=z_left.device)
    n_mc = member_l.numel()
    for m in range(n_mc):
        left = decoder.mean(z_left, int(member_l[m].item()))
        right = decoder.mean(z_right, int(member_k[m].item()))
        sq_sum = sq_sum + ((left - right) ** 2).sum()
    return sq_sum / n_mc


def compute_ensemble_geodesic(
    x_start,
    x_end,
    decoder,
    n_points=10,
    n_steps=200,
    mc_samples=8,
):
    """
    Compute a geodesic under the ensemble model-average curve energy:
    E(c) ~= sum_i E_{l,k}[||f_l(c(t_i)) - f_k(c(t_{i+1}))||^2].

    Parameters:
    x_start: [torch.Tensor]
        Starting point in latent space, shape (M,).
    x_end: [torch.Tensor]
        Ending point in latent space, shape (M,).
    decoder: [EnsembleGaussianDecoder-like]
        Decoder ensemble exposing `num_decoders` and `mean(z, member_idx)`.
    n_points: [int]
        Number of points along the discretized curve (including endpoints).
    n_steps: [int]
        Number of LBFGS optimization steps.
    mc_samples: [int]
        Number of Monte Carlo decoder-pair samples per segment.

    Returns:
    full_curve: [torch.Tensor]
        Optimized curve in latent space, shape (n_points, M).
    """
    if not hasattr(decoder, "num_decoders") or not hasattr(decoder, "mean"):
        raise ValueError("decoder must expose `num_decoders` and `mean(z, member_idx)`")

    t = torch.linspace(0, 1, n_points, device=x_start.device)
    curve = torch.stack([(1 - ti) * x_start + ti * x_end for ti in t])

    interior = curve[1:-1].clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([interior], lr=1)

    num_segments = n_points - 1
    member_l, member_k = _sample_member_pairs(
        num_segments=num_segments,
        mc_samples=mc_samples,
        num_decoders=decoder.num_decoders,
        device=x_start.device,
    )

    def energy_loss():
        optimizer.zero_grad()
        full_curve = torch.cat(
            [
                x_start.unsqueeze(0),
                interior,
                x_end.unsqueeze(0),
            ]
        )

        energy = torch.zeros((), device=full_curve.device)
        for i in range(num_segments):
            z_left = full_curve[i].unsqueeze(0)
            z_right = full_curve[i + 1].unsqueeze(0)
            energy = energy + _ensemble_segment_sqnorm(
                z_left,
                z_right,
                decoder,
                member_l[i],
                member_k[i],
            )

        energy.backward()
        return energy

    for _ in range(n_steps):
        optimizer.step(energy_loss)

    with torch.no_grad():
        full_curve = torch.cat(
            [
                x_start.unsqueeze(0),
                interior,
                x_end.unsqueeze(0),
            ]
        )
    return full_curve


def ensemble_curve_length(curve, decoder, mc_samples=32):
    """
    Approximate ensemble geodesic length as:
    L(c) ~= sum_i sqrt(E_{l,k}[||f_l(c(t_i)) - f_k(c(t_{i+1}))||^2]).

    Parameters:
    curve: [torch.Tensor]
        Discrete curve in latent space, shape (n_points, M).
    decoder: [EnsembleGaussianDecoder-like]
        Decoder ensemble exposing `num_decoders` and `mean(z, member_idx)`.
    mc_samples: [int]
        Number of Monte Carlo decoder-pair samples per segment.

    Returns:
    length: [float]
    """
    if not hasattr(decoder, "num_decoders") or not hasattr(decoder, "mean"):
        raise ValueError("decoder must expose `num_decoders` and `mean(z, member_idx)`")

    with torch.no_grad():
        num_segments = curve.shape[0] - 1
        member_l, member_k = _sample_member_pairs(
            num_segments=num_segments,
            mc_samples=mc_samples,
            num_decoders=decoder.num_decoders,
            device=curve.device,
        )

        seg_lengths = []
        for i in range(num_segments):
            z_left = curve[i].unsqueeze(0)
            z_right = curve[i + 1].unsqueeze(0)
            sqnorm = _ensemble_segment_sqnorm(
                z_left,
                z_right,
                decoder,
                member_l[i],
                member_k[i],
            )
            seg_lengths.append(torch.sqrt(sqnorm))

        return torch.stack(seg_lengths).sum().item()
