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
    
    For continuous actions (action_dim=1): outputs tanh-scaled value
    For discrete actions (action_dim>1): outputs logits
    """

    def __init__(self, obs_dim: int, z_dim: int, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(obs_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_head = nn.Linear(hidden_dim, action_dim)
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


class TubeRefiner(nn.Module):
    """
    Iterative refinement module for tube predictions.
    Takes current tube state (knots + stop_logit) and situation/commitment,
    outputs delta updates for refinement.

    This implements internal "reasoning" computation.
    """

    def __init__(
        self, state_dim: int, z_dim: int, M: int, hidden_dim: int = 64, pred_dim: int = None
    ):
        super().__init__()
        self.state_dim = state_dim  # input conditioning dimension
        self.pred_dim = pred_dim if pred_dim is not None else state_dim  # output prediction dimension
        self.M = M

        # Input: [s0, z, mu_knots_flat, logsig_knots_flat, stop_logit]
        # Note: mu_knots and logsig_knots are pred_dim * M each
        input_dim = state_dim + z_dim + (2 * self.pred_dim * M) + 1

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output: deltas for [mu_knots, logsig_knots, stop_logit]
        output_dim = (2 * self.pred_dim * M) + 1
        self.delta_head = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        s0: torch.Tensor,
        z: torch.Tensor,
        mu_knots: torch.Tensor,
        logsig_knots: torch.Tensor,
        stop_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute refinement deltas.

        Args:
            s0: [B, state_dim]
            z: [B, z_dim]
            mu_knots: [M, D] or [B, M, D]
            logsig_knots: [M, D] or [B, M, D]
            stop_logit: scalar or [B]

        Returns:
            delta_mu: [M, D] or [B, M, D]
            delta_logsig: [M, D] or [B, M, D]
            delta_stop: scalar or [B]
        """
        # Handle batch dimensions
        if mu_knots.dim() == 2:  # [M, D]
            mu_flat = mu_knots.flatten().unsqueeze(0)  # [1, M*D]
            logsig_flat = logsig_knots.flatten().unsqueeze(0)  # [1, M*D]
            stop_logit_input = stop_logit.unsqueeze(0).unsqueeze(0)  # [1, 1]
            unbatched = True
        else:  # [B, M, D]
            B = mu_knots.size(0)
            mu_flat = mu_knots.view(B, -1)  # [B, M*D]
            logsig_flat = logsig_knots.view(B, -1)  # [B, M*D]
            stop_logit_input = stop_logit.unsqueeze(-1)  # [B, 1]
            unbatched = False

        # Concatenate all inputs
        x = torch.cat([s0, z, mu_flat, logsig_flat, stop_logit_input], dim=-1)

        # Forward pass
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        deltas = self.delta_head(h)  # [B, output_dim]

        # Split deltas back into components
        D, M = self.pred_dim, self.M  # Use pred_dim for output dimension
        delta_mu_flat = deltas[:, : D * M]
        delta_logsig_flat = deltas[:, D * M : 2 * D * M]
        delta_stop = deltas[:, -1]

        # Reshape
        if unbatched:
            delta_mu = delta_mu_flat.squeeze(0).view(M, D)
            delta_logsig = delta_logsig_flat.squeeze(0).view(M, D)
            delta_stop = delta_stop.squeeze(0)
        else:
            delta_mu = delta_mu_flat.view(B, M, D)
            delta_logsig = delta_logsig_flat.view(B, M, D)

        return delta_mu, delta_logsig, delta_stop


class PonderHead(nn.Module):
    """
    Predicts reasoning stop probability p_refine_stop given (s0, z).
    Used to sample number of refinement steps via geometric distribution.
    """

    def __init__(self, state_dim: int, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, s0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s0: [B, state_dim]
            z: [B, z_dim]

        Returns:
            p_refine_stop: [B] or scalar, in (0, 1)
        """
        x = torch.cat([s0, z], dim=-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        stop_logit = self.stop_head(h).squeeze(-1)
        p_refine_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        return p_refine_stop


class DynamicPonderHead(nn.Module):
    """
    Dynamic pondering: decides whether to stop based on current cone volume
    and its derivative (improvement rate).

    Intuition:
    - High delta_vol: cone is tightening rapidly, keep thinking
    - Low delta_vol: cone has stabilized, stop to save compute
    """

    def __init__(self, state_dim: int, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Input: state, z, current_volume, volume_derivative
        self.fc1 = nn.Linear(state_dim + z_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(
        self,
        s0: torch.Tensor,
        z: torch.Tensor,
        current_vol: torch.Tensor,
        delta_vol: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s0: [B, state_dim]
            z: [B, z_dim]
            current_vol: [B] or scalar - current cone volume
            delta_vol: [B] or scalar - improvement rate (prev_vol - current_vol)

        Returns:
            p_stop: [B] or scalar - probability of stopping
        """
        # Ensure volume features are properly shaped
        if current_vol.dim() == 0:
            current_vol = current_vol.unsqueeze(0)
        if delta_vol.dim() == 0:
            delta_vol = delta_vol.unsqueeze(0)

        # Expand to batch size if needed
        if s0.size(0) > 1 and current_vol.size(0) == 1:
            current_vol = current_vol.expand(s0.size(0))
            delta_vol = delta_vol.expand(s0.size(0))

        vol_feats = torch.stack([current_vol, delta_vol], dim=-1)

        x = torch.cat([s0, z, vol_feats], dim=-1)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))

        # Output probability of STOPPING
        stop_logit = self.stop_head(h).squeeze(-1)
        p_stop = torch.sigmoid(stop_logit).clamp(1e-4, 1.0 - 1e-4)
        return p_stop
