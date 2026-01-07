"""
Neural network architectures for continuous port TAM.
"""

from typing import Tuple

import torch
import torch.nn as nn


class ActorNet(nn.Module):
    """
    q(z | s0): outputs mean and log-std for a Gaussian latent port z.
    This is the continuous port selector that maps situations to commitments.
    """

    def __init__(self, state_dim: int, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logstd_head = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        s0: [B, state_dim]
        returns: z_mu [B, z_dim], z_logstd [B, z_dim]
        """
        h = self.relu(self.fc1(s0))
        h = self.relu(self.fc2(h))
        z_mu = self.mu_head(h)
        z_logstd = self.logstd_head(h).clamp(-6.0, 2.0)
        return z_mu, z_logstd


class SharedPolicy(nn.Module):
    """
    Policy network: action = pi(state, z)
    Conditioned on latent commitment z.
    """

    def __init__(self, state_dim: int, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()


class SharedTube(nn.Module):
    """
    Tube prediction network: predicts knots for mu(t), log_sigma(t), and stop prob.
    Input: [s0, z]
    Output: vec of size state_dim*M + state_dim*M + 1
    """

    def __init__(
        self, state_dim: int, z_dim: int, hidden_dim: int = 64, out_dim: int = 1
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_head = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
