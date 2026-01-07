"""
Utility functions for TAM training.
"""

from typing import Tuple

import numpy as np
import torch


def gaussian_nll(
    mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Elementwise Gaussian NLL up to constant. Shapes [*, D]. Differentiable."""
    var = torch.exp(log_var)
    sq = (target - mu) ** 2
    return 0.5 * (log_var + sq / (var + 1e-8))


def kl_diag_gaussian_to_standard(
    mu: torch.Tensor, logstd: torch.Tensor
) -> torch.Tensor:
    """
    KL(N(mu,diag(std^2)) || N(0,I)) per batch element.
    mu: [B, D]
    logstd: [B, D]
    returns: [B]
    """
    var = torch.exp(2.0 * logstd)
    return 0.5 * torch.sum(var + mu**2 - 1.0 - torch.log(var + 1e-8), dim=-1)


def truncated_geometric_weights(
    p_stop: torch.Tensor, T: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Option-3 style weights for an observed horizon T with a geometric stop prob p_stop.
    p_stop: scalar tensor in (0,1)
    Returns:
      w: [T] weights summing to 1 (with tail mass forced into step T)
      E_T: scalar tensor expected step index under w over 1..T
    """
    # survival s_t = (1-p)^(t-1)
    t_idx = torch.arange(1, T + 1, device=p_stop.device, dtype=p_stop.dtype)
    one_minus = (1.0 - p_stop).clamp(1e-6, 1.0 - 1e-6)

    s = one_minus ** (t_idx - 1.0)  # [T]
    w = s * p_stop  # [T], unnormalized "stop at t"

    # tail mass beyond T gets added to last step
    tail = one_minus**T  # prob not stopped by step T
    w = w.clone()
    w[-1] = w[-1] + tail

    w = w / (w.sum() + 1e-8)
    E_T = (w * t_idx).sum()
    return w, E_T


def sample_truncated_geometric(p_stop: float, maxH: int, minT: int = 1) -> int:
    """
    Sample T in [1..maxH] from truncated geometric with parameter p_stop.
    Enforce minT by collapsing probability mass below minT into minT.
    """
    p = float(np.clip(p_stop, 1e-4, 1.0 - 1e-4))
    one_minus = 1.0 - p
    # P(T=t) = (1-p)^(t-1) p for t<maxH, and tail mass at maxH
    probs = np.array(
        [(one_minus ** (t - 1)) * p for t in range(1, maxH)], dtype=np.float64
    )
    tail = one_minus ** (maxH - 1)
    probs = np.concatenate(
        [probs, np.array([tail], dtype=np.float64)], axis=0
    )  # len=maxH

    probs = probs / (probs.sum() + 1e-12)

    if minT > 1:
        m = float(np.sum(probs[: minT - 1]))
        probs[minT - 1] += m
        probs[: minT - 1] = 0.0
        probs = probs / (probs.sum() + 1e-12)

    return int(np.random.choice(np.arange(1, maxH + 1), p=probs))


def interp1d_linear(knots: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation from knot values.
    knots: [M, D]
    t: [T] in [0,1]
    returns: [T, D]
    """
    M, D = knots.shape
    # map t to continuous knot index in [0, M-1]
    u = t * (M - 1)
    i0 = torch.floor(u).long().clamp(0, M - 2)  # [T]
    i1 = i0 + 1  # [T]
    w = (u - i0.float()).unsqueeze(-1)  # [T, 1]

    v0 = knots[i0]  # [T, D]
    v1 = knots[i1]  # [T, D]
    return (1 - w) * v0 + w * v1
