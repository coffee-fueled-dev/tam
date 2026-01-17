import torch
import numpy as np

class CausalSpline:
    @staticmethod
    def interpolate(knots, sigmas, resolution=40):
        """
        knots: (B, K, 3)
        sigmas: (B, K, 1) - Learned radius at each knot
        """
        B, K, D = knots.shape
        device = knots.device
        
        # Safety check: need at least 2 knots for interpolation
        if K < 2:
            # Return just the start and end points
            start = knots[:, 0:1, :]  # (B, 1, 3)
            end = knots[:, -1:, :] if K > 1 else knots[:, 0:1, :]
            trajectory = torch.cat([start, end], dim=1)
            sigma_traj = torch.cat([sigmas[:, 0:1, :], sigmas[:, -1:, :] if K > 1 else sigmas[:, 0:1, :]], dim=1)
            return trajectory, sigma_traj
        
        # We concatenate knots and sigmas to interpolate them together
        # Combined state: [x, y, z, sigma]
        combined = torch.cat([knots, sigmas], dim=-1) # (B, K, 4)

        # Ghost knots for C-R Spline
        p0 = 2 * combined[:, 0:1] - combined[:, 1:2]
        pn = 2 * combined[:, -1:] - combined[:, -2:-1]
        padded = torch.cat([p0, combined, pn], dim=1)

        points_per_seg = max(4, int(np.ceil(resolution / max(1, K - 1))))
        t = torch.linspace(0, 1, points_per_seg, device=device)[:-1]
        t1, t2, t3 = t, t**2, t**3

        all_segments = []
        for i in range(1, K):
            P0, P1, P2, P3 = [padded[:, i+j].unsqueeze(1) for j in range(-1, 3)]
            
            seg = 0.5 * (2*P1 + (-P0 + P2)*t1.view(1,-1,1) + 
                         (2*P0 - 5*P1 + 4*P2 - P3)*t2.view(1,-1,1) + 
                         (-P0 + 3*P1 - 3*P2 + P3)*t3.view(1,-1,1))
            all_segments.append(seg)

        # Add endpoint
        all_segments.append(combined[:, -1:].view(B, 1, 4))
        full_trajectory = torch.cat(all_segments, dim=1)
        
        return full_trajectory[:, :, :3], full_trajectory[:, :, 3:]

