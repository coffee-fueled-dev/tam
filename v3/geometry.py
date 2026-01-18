import torch
import numpy as np

class CausalSpline:
    """
    Dimension-agnostic linear interpolator.
    
    Works with any state dimension - linearly connects knots without creating arcs.
    This allows the Actor to learn smoothness through knot placement rather than
    having it enforced by interpolation.
    """
    @staticmethod
    def interpolate(knots, sigmas, resolution=40):
        """
        Interpolate knots into a continuous trajectory using linear interpolation.
        
        No arcs or curves - just straight lines between knots. This allows the Actor
        to learn smoothness through knot placement if beneficial, but doesn't force
        smoothness when sharp turns are acceptable.
        
        Args:
            knots: (B, K, state_dim) - Knot positions in state space
            sigmas: (B, K, 1) - Learned radius/precision at each knot
            resolution: Number of interpolation points per segment
            
        Returns:
            trajectory: (B, T, state_dim) - Interpolated trajectory
            sigma_traj: (B, T, 1) - Interpolated sigma values
        """
        B, K, state_dim = knots.shape
        device = knots.device
        
        # Safety check: need at least 2 knots for interpolation
        if K < 2:
            # Return just the start and end points
            start = knots[:, 0:1, :]  # (B, 1, state_dim)
            end = knots[:, -1:, :] if K > 1 else knots[:, 0:1, :]
            trajectory = torch.cat([start, end], dim=1)
            sigma_traj = torch.cat([sigmas[:, 0:1, :], sigmas[:, -1:, :] if K > 1 else sigmas[:, 0:1, :]], dim=1)
            return trajectory, sigma_traj
        
        # Linear interpolation: connect knots with straight lines
        # Calculate points per segment
        points_per_seg = max(2, int(np.ceil(resolution / max(1, K - 1))))
        
        all_segments = []
        all_sigmas = []
        
        for i in range(K - 1):
            # Get start and end knots for this segment
            k_start = knots[:, i:i+1, :]  # (B, 1, state_dim)
            k_end = knots[:, i+1:i+2, :]  # (B, 1, state_dim)
            
            # Get start and end sigmas for this segment
            s_start = sigmas[:, i:i+1, :]  # (B, 1, 1)
            s_end = sigmas[:, i+1:i+2, :]  # (B, 1, 1)
            
            # Linear interpolation parameter: t goes from 0 to 1
            t = torch.linspace(0, 1, points_per_seg, device=device)
            
            # Interpolate knots: (1-t) * k_start + t * k_end
            # Shape: (B, points_per_seg, state_dim)
            segment = (1 - t.view(1, -1, 1)) * k_start + t.view(1, -1, 1) * k_end
            
            # Interpolate sigmas: (1-t) * s_start + t * s_end
            # Shape: (B, points_per_seg, 1)
            sigma_segment = (1 - t.view(1, -1, 1)) * s_start + t.view(1, -1, 1) * s_end
            
            all_segments.append(segment)
            all_sigmas.append(sigma_segment)
        
        # Add final endpoint (last knot)
        all_segments.append(knots[:, -1:, :])  # (B, 1, state_dim)
        all_sigmas.append(sigmas[:, -1:, :])  # (B, 1, 1)
        
        # Concatenate all segments
        trajectory = torch.cat(all_segments, dim=1)  # (B, T, state_dim)
        sigma_traj = torch.cat(all_sigmas, dim=1)  # (B, T, 1)
        
        return trajectory, sigma_traj

