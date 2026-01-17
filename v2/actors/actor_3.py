import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os
import hashlib
from datetime import datetime
from collections import defaultdict


# ==========================================
# 1. TKN: SPATIAL PATTERN TOKENIZER
# ==========================================

class TknHead:
    """
    Stateless head for discovering patterns in quantized spatial data.
    
    Implements inclusion/boundary logic: patterns grow until hitting novelty,
    then emit. No dynamic dictionary - patterns are hashed to stable IDs.
    
    Exposes lattice traits: buffer hash, hub-ness (confidence), and surprise.
    """
    def __init__(self, name, quantization_bins=11, quant_range=(-2.0, 2.0), vocab_size=65536):
        self.name = name
        self.quantization_bins = quantization_bins
        self.quant_range = quant_range
        self.vocab_size = vocab_size
        self.buffer = []  # Current pattern being built
        self.pattern = ""  # Current pattern string (for boundary detection)
        self.seen_patterns = set()  # Only for tracking novelty (surprise detection)
        self.pattern_counts = defaultdict(int)  # Track frequency for hub-ness
        self.last_step_was_emission = False  # Track boundary events
        
    def quantize(self, value, min_val=None, max_val=None):
        """Quantize a float value into discrete bins."""
        min_val = min_val if min_val is not None else self.quant_range[0]
        max_val = max_val if max_val is not None else self.quant_range[1]
        clamped = np.clip(value, min_val, max_val)
        bin_size = (max_val - min_val) / self.quantization_bins
        quantized = int((clamped - min_val) / bin_size)
        return quantized
    
    def process(self, quantized_input):
        """
        Process a quantized input value using inclusion/boundary logic.
        
        Args:
            quantized_input: Quantized integer value
            
        Returns:
            tuple: (emitted_pattern, is_novel)
                - emitted_pattern: List of quantized values if pattern emitted, None otherwise
                - is_novel: True if this is the first time seeing this pattern
        """
        char_input = str(quantized_input)
        extended = self.pattern + "|" + char_input if self.pattern else char_input
        
        # Check if extended pattern has been seen before
        is_novel = extended not in self.seen_patterns
        self.last_step_was_emission = False
        
        if not is_novel:
            # Inclusion: Grow the pattern (seen before, continue)
            self.pattern = extended
            self.buffer.append(quantized_input)
            return None, False  # No token emitted yet
        else:
            # Boundary: Emit sequence (novel pattern detected)
            emitted = list(self.buffer) if self.buffer else [quantized_input]
            
            # Mark this pattern as seen and track frequency (for hub-ness)
            if self.pattern:
                self.seen_patterns.add(extended)
                pattern_key = self.pattern  # The pattern that was completed
                self.pattern_counts[pattern_key] += 1
            
            self.last_step_was_emission = True
            
            # Start new pattern
            self.buffer = [quantized_input]
            self.pattern = char_input
            
            return emitted, is_novel  # Emit the discovered pattern
    
    def get_head_state(self):
        """
        Get lattice traits: buffer hash, hub-ness, and surprise.
        
        Returns:
            tuple: (buffer_hash, hub_count, surprise)
                - buffer_hash: Hash of current growing buffer (eager signal)
                - hub_count: Frequency of current pattern (confidence/precision)
                - surprise: 1.0 if last step was emission (boundary), else 0.0
        """
        # Buffer Hash: The 'eager' signal - current activity before emission
        buffer_str = "-".join(map(str, self.buffer)) if self.buffer else ""
        if buffer_str:
            buffer_hash = int(hashlib.md5(buffer_str.encode()).hexdigest(), 16) % self.vocab_size
        else:
            buffer_hash = 0
        
        # Hub-ness: Frequency of current pattern in the dictionary
        # High frequency = High Confidence = Low Sigma (precision weighting)
        pattern_key = self.pattern if self.pattern else ""
        hub_count = self.pattern_counts.get(pattern_key, 0)
        
        # Surprise: 1.0 if we just emitted/hit a boundary, else 0.0
        # This is the information-theoretic surprise signal
        surprise = 1.0 if self.last_step_was_emission else 0.0
        
        return buffer_hash, hub_count, surprise


class InvariantLattice:
    """
    Stateless geometric tokenizer that maps patterns to invariant token IDs.
    
    Uses stable hashing to ensure any specific physical maneuver always
    maps to the same token ID across all time and episodes.
    """
    def __init__(self, head_names, vocab_size=65536):
        self.head_names = head_names
        self.vocab_size = vocab_size
        # Current token per head (persistent state)
        self.current_lattice = torch.zeros(len(head_names), dtype=torch.long)
        # Track novel patterns for surprise detection
        self.novel_patterns = set()
        
    def update(self, head_emissions):
        """
        Update lattice with pattern emissions from heads.
        
        Args:
            head_emissions: dict mapping head_name -> (pattern_list, is_novel) or None
            
        Returns:
            current_lattice: (num_heads,) tensor of token IDs
            surprise_mask: (num_heads,) bool tensor indicating novel patterns
        """
        surprise_mask = torch.zeros(len(self.head_names), dtype=torch.bool)
        
        for i, name in enumerate(self.head_names):
            emission_data = head_emissions.get(name)
            if emission_data is not None:
                pattern, is_novel = emission_data
                if pattern is not None:
                    # Stable hashing: Geometry -> Invariant ID
                    pattern_str = f"{name}:" + "-".join(map(str, pattern))
                    hash_hex = hashlib.md5(pattern_str.encode()).hexdigest()
                    token_id = int(hash_hex, 16) % self.vocab_size
                    self.current_lattice[i] = token_id
                    
                    # Track novelty for surprise mechanism
                    if is_novel:
                        surprise_mask[i] = True
                        self.novel_patterns.add(pattern_str)
        
        return self.current_lattice.clone(), surprise_mask
    
    def reset_episode(self):
        """Reset episode state (but keep lattice tokens - they're invariant)."""
        pass  # Lattice tokens persist across episodes


class TknProcessor:
    """
    Multi-head processor for quantizing geometric data into higher-level motifs.
    
    Manages parallel heads for:
    - X, Y, Z delta quantization (movement direction)
    - Proximity quantization (distance to nearest obstacle)
    - Goal direction quantization (X, Y, Z components)
    """
    def __init__(self, quantization_bins=11, quant_range=(-2.0, 2.0), vocab_size=65536):
        self.quantization_bins = quantization_bins
        self.quant_range = quant_range
        self.vocab_size = vocab_size
        self.heads = {
            "delta_x": TknHead("delta_x", quantization_bins, quant_range, vocab_size),
            "delta_y": TknHead("delta_y", quantization_bins, quant_range, vocab_size),
            "delta_z": TknHead("delta_z", quantization_bins, quant_range, vocab_size),
            "proximity": TknHead("proximity", quantization_bins, quant_range=(-5.0, 10.0), vocab_size=vocab_size),
            "goal_dir_x": TknHead("goal_dir_x", quantization_bins, quant_range, vocab_size),
            "goal_dir_y": TknHead("goal_dir_y", quantization_bins, quant_range, vocab_size),
            "goal_dir_z": TknHead("goal_dir_z", quantization_bins, quant_range, vocab_size),
        }
        self.head_names = list(self.heads.keys())
        self.previous_pos = None
        self.previous_rel_goal = None
        
        # Initialize invariant lattice
        self.lattice = InvariantLattice(self.head_names, vocab_size=vocab_size)
        
    def process_observation(self, current_pos, raw_obs, obstacles):
        """
        Process a raw observation through tkn heads and update invariant lattice.
        
        Args:
            current_pos: Current position (3,) tensor
            raw_obs: Raw observation tensor (raw_ctx_dim,)
            obstacles: List of (position, radius) tuples
            
        Returns:
            dict with:
                - lattice_tokens: (num_heads,) tensor of token IDs
                - surprise_mask: (num_heads,) bool tensor indicating novel patterns
                - lattice_traits: (num_heads, 2) tensor of [hub_count, surprise] per head
                - buffer_hashes: (num_heads,) tensor of buffer hash IDs
                - metadata: Dict with quantized values, emissions, etc.
        """
        current_pos_np = current_pos.squeeze().cpu().numpy() if isinstance(current_pos, torch.Tensor) else current_pos
        
        # Extract relative goal from raw_obs (first 3 elements)
        rel_goal = raw_obs[:3].cpu().numpy() if isinstance(raw_obs, torch.Tensor) else raw_obs[:3]
        
        # Calculate deltas (change in position)
        if self.previous_pos is not None:
            delta = current_pos_np - self.previous_pos
        else:
            delta = np.zeros(3)
        
        # Quantize and process deltas
        delta_x_q = self.heads["delta_x"].quantize(delta[0])
        delta_y_q = self.heads["delta_y"].quantize(delta[1])
        delta_z_q = self.heads["delta_z"].quantize(delta[2])
        
        delta_x_pattern, delta_x_novel = self.heads["delta_x"].process(delta_x_q)
        delta_y_pattern, delta_y_novel = self.heads["delta_y"].process(delta_y_q)
        delta_z_pattern, delta_z_novel = self.heads["delta_z"].process(delta_z_q)
        
        # Calculate proximity to nearest obstacle
        min_proximity = float('inf')
        if obstacles:
            for obs_p, obs_r in obstacles:
                obs_pos = np.array(obs_p)
                distance = np.linalg.norm(current_pos_np - obs_pos) - obs_r
                min_proximity = min(min_proximity, distance)
        else:
            min_proximity = 10.0  # No obstacles
        
        # Quantize and process proximity
        proximity_q = self.heads["proximity"].quantize(min_proximity)
        proximity_pattern, proximity_novel = self.heads["proximity"].process(proximity_q)
        
        # Quantize and process goal direction
        goal_dir_x_q = self.heads["goal_dir_x"].quantize(rel_goal[0])
        goal_dir_y_q = self.heads["goal_dir_y"].quantize(rel_goal[1])
        goal_dir_z_q = self.heads["goal_dir_z"].quantize(rel_goal[2])
        
        goal_dir_x_pattern, goal_dir_x_novel = self.heads["goal_dir_x"].process(goal_dir_x_q)
        goal_dir_y_pattern, goal_dir_y_novel = self.heads["goal_dir_y"].process(goal_dir_y_q)
        goal_dir_z_pattern, goal_dir_z_novel = self.heads["goal_dir_z"].process(goal_dir_z_q)
        
        # Collect all emissions with novelty flags
        head_emissions = {
            "delta_x": (delta_x_pattern, delta_x_novel) if delta_x_pattern is not None else None,
            "delta_y": (delta_y_pattern, delta_y_novel) if delta_y_pattern is not None else None,
            "delta_z": (delta_z_pattern, delta_z_novel) if delta_z_pattern is not None else None,
            "proximity": (proximity_pattern, proximity_novel) if proximity_pattern is not None else None,
            "goal_dir_x": (goal_dir_x_pattern, goal_dir_x_novel) if goal_dir_x_pattern is not None else None,
            "goal_dir_y": (goal_dir_y_pattern, goal_dir_y_novel) if goal_dir_y_pattern is not None else None,
            "goal_dir_z": (goal_dir_z_pattern, goal_dir_z_novel) if goal_dir_z_pattern is not None else None,
        }
        
        # Update invariant lattice
        lattice_tokens, surprise_mask = self.lattice.update(head_emissions)
        
        # Collect lattice traits from all heads (epistemic status)
        buffer_hashes = []
        hub_counts = []
        surprises = []
        
        for head_name in self.head_names:
            buffer_hash, hub_count, surprise = self.heads[head_name].get_head_state()
            buffer_hashes.append(buffer_hash)
            hub_counts.append(hub_count)
            surprises.append(surprise)
        
        # Convert to tensors
        buffer_hashes_tensor = torch.tensor(buffer_hashes, dtype=torch.long)  # (num_heads,)
        lattice_traits = torch.stack([
            torch.tensor(hub_counts, dtype=torch.float32),  # Hub-ness (confidence)
            torch.tensor(surprises, dtype=torch.float32)     # Surprise (boundary)
        ], dim=1)  # (num_heads, 2)
        
        # Update state
        self.previous_pos = current_pos_np.copy()
        self.previous_rel_goal = rel_goal.copy()
        
        # Build output
        return {
            "lattice_tokens": lattice_tokens,  # (num_heads,) tensor of token IDs
            "surprise_mask": surprise_mask,  # (num_heads,) bool tensor
            "lattice_traits": lattice_traits,  # (num_heads, 2) tensor [hub_count, surprise]
            "buffer_hashes": buffer_hashes_tensor,  # (num_heads,) tensor of buffer hash IDs
            "rel_goal": torch.tensor(rel_goal, dtype=torch.float32),  # (3,) high-fidelity intent
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "current_pos": current_pos_np.tolist(),
                "rel_goal": rel_goal.tolist(),
                "delta": delta.tolist(),
                "min_proximity": float(min_proximity),
                "quantized": {
                    "delta_x": int(delta_x_q),
                    "delta_y": int(delta_y_q),
                    "delta_z": int(delta_z_q),
                    "proximity": int(proximity_q),
                    "goal_dir_x": int(goal_dir_x_q),
                    "goal_dir_y": int(goal_dir_y_q),
                    "goal_dir_z": int(goal_dir_z_q),
                },
                "emissions": {k: (v[0] if v is not None else None) 
                             for k, v in head_emissions.items()},
                "has_emissions": any(v is not None for v in head_emissions.values()),
                "has_surprise": surprise_mask.any().item() if isinstance(surprise_mask, torch.Tensor) else any(surprise_mask),
                "lattice_traits": {
                    "buffer_hashes": buffer_hashes,
                    "hub_counts": hub_counts,
                    "surprises": surprises
                }
            }
        }
    
    def reset_episode(self):
        """Reset state at start of new episode."""
        self.previous_pos = None
        self.previous_rel_goal = None
        self.lattice.reset_episode()


# ==========================================
# 2. INFERENCE ENGINE: TOKEN-BASED WORLD MODEL
# ==========================================

class HybridInferenceEngine(nn.Module):
    """
    Hybrid Inference Engine that processes:
    1. Discrete tokens (what) - geometric pattern IDs
    2. Lattice traits (epistemic status) - hub-ness, surprise, buffer state
    3. High-fidelity intent (where) - raw rel_goal vector
    
    This creates a rich "Full Situation" representation that combines:
    - Semantic meaning (token embeddings)
    - Epistemic confidence (lattice topology)
    - Spatial precision (direct goal vector)
    """
    def __init__(self, num_heads, vocab_size=65536, token_embed_dim=16, latent_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.token_embed_dim = token_embed_dim
        self.latent_dim = latent_dim
        
        # Token Stream: Discrete pattern IDs -> Dense embeddings
        self.token_embedding = nn.Embedding(vocab_size, token_embed_dim)
        
        # Lattice Traits Stream: Hub-ness + Surprise -> Features
        # (num_heads, 2) -> (num_heads * 2) -> 32
        self.trait_fc = nn.Sequential(
            nn.Linear(num_heads * 2, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # Intent Stream: High-fidelity rel_goal (3,) -> Features
        self.intent_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # Combined GRU input: (Token Embeds) + Traits + Intent
        # (num_heads * token_embed_dim) + 32 + 32
        total_input_dim = (num_heads * token_embed_dim) + 32 + 32
        self.gru = nn.GRUCell(total_input_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        
    def forward(self, lattice_tokens, lattice_traits, rel_goal, h_prev):
        """
        Convert geometric tokens + lattice traits + intent to latent situation.
        
        Args:
            lattice_tokens: (B, num_heads) tensor of token IDs
            lattice_traits: (B, num_heads, 2) tensor of [hub_count, surprise] per head
            rel_goal: (B, 3) tensor of high-fidelity relative goal vector
            h_prev: (B, latent_dim) previous hidden state (situation from last step)
        
        Returns:
            h_next: (B, latent_dim) new hidden state (current situation x_n)
        """
        # 1. Token Stream: Embed discrete pattern IDs
        token_embeds = self.token_embedding(lattice_tokens)  # (B, num_heads, token_embed_dim)
        token_flat = token_embeds.view(token_embeds.size(0), -1)  # (B, num_heads * token_embed_dim)
        
        # 2. Lattice Traits Stream: Process epistemic status
        trait_flat = lattice_traits.view(lattice_traits.size(0), -1)  # (B, num_heads * 2)
        trait_feat = self.trait_fc(trait_flat)  # (B, 32)
        
        # 3. Intent Stream: Process high-fidelity goal vector
        intent_feat = self.intent_fc(rel_goal)  # (B, 32)
        
        # 4. Concatenate all channels (The 'Full Situation')
        combined = torch.cat([token_flat, trait_feat, intent_feat], dim=-1)  # (B, total_input_dim)
        
        # 5. Update latent situation x_n
        h_next = self.gru(combined, h_prev)
        return self.ln(h_next)


# ==========================================
# 3. GEOMETRY UTILITIES: CATMULL-ROM SPLINE
# ==========================================

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


# ==========================================
# 4. THE ACTOR: CAUSAL KNOT ENGINE (LATENT-AGNOSTIC)
# ==========================================

class Actor(nn.Module):
    """
    Actor operates on latent situations from InferenceEngine.
    
    The Actor no longer cares where its input comes from - it's agnostic to
    whether the latent representation encodes spatial features, temporal patterns,
    or any other world structure. It simply proposes affordance tubes based on
    the latent situation and intent.
    """
    def __init__(self, latent_dim, intent_dim, n_ports=4, n_knots=6, interp_res=40):
        super().__init__()
        self.n_ports = n_ports
        self.n_knots = n_knots
        self.interp_res = interp_res
        
        # Encoder: processes latent situation + intent
        # Actor doesn't perform spatial calculations - it operates in latent space
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim + intent_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256)
        )
        
        # Port Proposer
        # Output size: M * (Logit + K*3 + K*1 + K*1) where last K*1 is knot existence logits
        # Note: K*3 are now RESIDUAL OFFSETS, not absolute positions
        self.port_head = nn.Linear(256, n_ports * (1 + n_knots * 5))
        
    def forward(self, latent_situation, intent, previous_velocity=None):
        """
        Args:
            latent_situation: (B, latent_dim) latent situation from InferenceEngine
            intent: (B, 3) intent/target direction tensor
            previous_velocity: Optional (B, 3) or (3,) tensor of previous path velocity for G1 continuity
                              If provided, enforces that first knot direction matches this velocity
        
        Returns:
            logits: (B, M)
            mu_t:   (B, M, T, 3) - Interpolated Tube
            sigma_t:(B, M, T, 1) - Interpolated Radius
            knot_mask: (B, M, K) - Knot existence mask (0.0 = inactive, 1.0 = active)
            knot_steps: (B, M, K, 3) - Residual offsets (for loss calculation)
        """
        B = latent_situation.size(0)
        situation = self.encoder(torch.cat([latent_situation, intent], dim=-1))
        
        raw_out = self.port_head(situation).view(B, self.n_ports, -1)
        logits = raw_out[:, :, 0]
        
        # Extract Knot Steps (RESIDUAL OFFSETS), Sigmas, and Knot Existence Logits
        geo_params = raw_out[:, :, 1:].view(B, self.n_ports, self.n_knots, 5)
        
        # RESIDUAL KNOT LOGIC: Predict relative offsets instead of absolute positions
        # Use tanh to bound the step size of each knot (prevents erratic snapping)
        knot_steps = torch.tanh(geo_params[..., :3]) * 2.0  # (B, M, K, 3) bounded to [-2, 2]
        
        sigmas_raw = F.softplus(geo_params[..., 3:4]) + 0.1
        knot_existence_logits = geo_params[..., 4:5]  # (B, M, K, 1)
        
        # Apply sigmoid to get knot existence mask (0.0 = inactive, 1.0 = active)
        # First and last knots are always active (required for spline)
        knot_mask_raw = torch.sigmoid(knot_existence_logits).squeeze(-1)  # (B, M, K)
        # Create mask with first and last always active (non-inplace operation)
        device = knot_mask_raw.device
        ones_first = torch.ones(B, self.n_ports, 1, device=device)
        ones_last = torch.ones(B, self.n_ports, 1, device=device)
        knot_mask = torch.cat([ones_first, knot_mask_raw[:, :, 1:-1], ones_last], dim=-1)
        
        # RESIDUAL KNOT CONSTRUCTION: The spline 'grows' from the origin (0,0,0)
        # Use cumsum to make each knot relative to the one before it
        knots_rel = torch.cumsum(knot_steps, dim=2)  # (B, M, K, 3)
        # Causal Anchoring: Ensure the first knot is always exactly at the start
        knots_rel = knots_rel - knots_rel[:, :, 0:1, :]
        
        # G1 CONTINUITY: Enforce by construction (not penalty)
        # If previous_velocity is provided, ensure first knot direction matches it
        # This prevents "jagged zags" structurally rather than through loss penalties
        if previous_velocity is not None:
            # Normalize previous velocity to get direction
            if previous_velocity.dim() == 1:
                prev_vel = previous_velocity.unsqueeze(0)  # (1, 3)
            else:
                prev_vel = previous_velocity  # (B, 3) or (1, 3)
            
            # Ensure prev_vel is (B, 3) for broadcasting
            if prev_vel.size(0) == 1 and B > 1:
                prev_vel = prev_vel.expand(B, -1)
            elif prev_vel.size(0) != B:
                prev_vel = prev_vel[:B]  # Take first B if needed
            
            prev_vel_norm = prev_vel / (torch.norm(prev_vel, dim=-1, keepdim=True) + 1e-6)  # (B, 3)
            
            # Project first knot direction onto previous velocity direction
            # knots_rel[:, :, 1, :] is the second knot (first after origin)
            # We want the direction from knot 0 to knot 1 to align with prev_vel_norm
            if self.n_knots > 1:
                # Get current first segment direction
                first_segment = knots_rel[:, :, 1:2, :]  # (B, M, 1, 3)
                first_segment_norm = first_segment / (torch.norm(first_segment, dim=-1, keepdim=True) + 1e-6)
                
                # Project onto previous velocity direction
                # We want: first_segment_dir = prev_vel_norm * scale
                # So: scale = dot(first_segment_dir, prev_vel_norm)
                # Then: first_segment = prev_vel_norm * scale * magnitude
                prev_vel_expanded = prev_vel_norm.view(B, 1, 1, 3)  # (B, 1, 1, 3) broadcasts to (B, M, 1, 3)
                
                # Get magnitude of first segment
                first_segment_mag = torch.norm(first_segment, dim=-1, keepdim=True)  # (B, M, 1, 1)
                
                # Align direction: use previous velocity direction, preserve magnitude
                # Blend: 80% previous velocity direction, 20% network's learned direction
                # This allows the network to learn while enforcing continuity
                aligned_direction = 0.8 * prev_vel_expanded + 0.2 * first_segment_norm
                aligned_direction = aligned_direction / (torch.norm(aligned_direction, dim=-1, keepdim=True) + 1e-6)
                aligned_first_segment = aligned_direction * first_segment_mag
                
                # Replace first segment with aligned version
                knots_rel = torch.cat([knots_rel[:, :, :1, :], aligned_first_segment, knots_rel[:, :, 2:, :]], dim=2)
        
        # INTENT BIAS: Push the last knot toward the intent
        # This ensures that even with random weights, tubes are biased toward the goal
        # Reshape intent to (B, 1, 3) to broadcast across ports (M dimension)
        intent_bias = intent.view(B, 1, 3)  # (B, 1, 3) broadcasts to (B, M, 3)
        # Add a fraction of intent to the last knot to guide the network (non-inplace)
        # Using 0.5 means the network learns to add the other 0.5 through training
        knots_rel_last = knots_rel[:, :, -1:, :] + intent_bias.unsqueeze(2) * 0.5
        knots_rel = torch.cat([knots_rel[:, :, :-1, :], knots_rel_last], dim=2)
        
        # Filter inactive knots: keep only knots with mask > 0.5
        # This creates variable-length knot sequences per port
        mu_dense_list = []
        sigma_dense_list = []
        
        for b in range(B):
            for m in range(self.n_ports):
                # Get active knots for this port
                active_mask = knot_mask[b, m] > 0.5  # (K,)
                active_indices = torch.where(active_mask)[0]
                device = active_indices.device if len(active_indices) > 0 else knots_rel.device
                
                # Ensure at least first and last knots are included
                if len(active_indices) == 0:
                    # If no knots active, use first and last
                    active_indices = torch.tensor([0, self.n_knots - 1], device=device)
                else:
                    # Ensure first knot is included
                    if not torch.any(active_indices == 0):
                        active_indices = torch.cat([torch.tensor([0], device=device), active_indices])
                    # Ensure last knot is included
                    if not torch.any(active_indices == self.n_knots - 1):
                        active_indices = torch.cat([active_indices, torch.tensor([self.n_knots - 1], device=device)])
                    # Sort and remove duplicates
                    active_indices = torch.unique(torch.sort(active_indices)[0])
                
                # Extract active knots and sigmas
                active_knots = knots_rel[b, m, active_indices, :].unsqueeze(0)  # (1, K_active, 3)
                active_sigmas = sigmas_raw[b, m, active_indices, :].unsqueeze(0)  # (1, K_active, 1)
                
                # Interpolate with filtered knots
                mu_dense_port, sigma_dense_port = CausalSpline.interpolate(
                    active_knots, active_sigmas, resolution=self.interp_res
                )
                
                mu_dense_list.append(mu_dense_port.squeeze(0))  # (T, 3)
                sigma_dense_list.append(sigma_dense_port.squeeze(0))  # (T, 1)
        
        # Pad to same length for batching (use max length)
        max_T = max(m.shape[0] for m in mu_dense_list)
        device = mu_dense_list[0].device
        
        mu_t_padded = []
        sigma_t_padded = []
        for mu, sigma in zip(mu_dense_list, sigma_dense_list):
            T = mu.shape[0]
            if T < max_T:
                # Pad with last value
                pad_mu = mu[-1:].repeat(max_T - T, 1)
                pad_sigma = sigma[-1:].repeat(max_T - T, 1)
                mu_padded = torch.cat([mu, pad_mu], dim=0)
                sigma_padded = torch.cat([sigma, pad_sigma], dim=0)
            else:
                mu_padded = mu
                sigma_padded = sigma
            mu_t_padded.append(mu_padded)
            sigma_t_padded.append(sigma_padded)
        
        # Stack and reshape
        mu_t = torch.stack(mu_t_padded).view(B, self.n_ports, max_T, 3)
        sigma_t = torch.stack(sigma_t_padded).view(B, self.n_ports, max_T, 1)
        
        return logits, mu_t, sigma_t, knot_mask, knot_steps

    def select_port(self, logits, mu_t, intent_target):
        """Select port based on closest tube endpoint to target, weighted by logits."""
        end_points = mu_t[:, :, -1, :] 
        dist = torch.norm(end_points - intent_target.unsqueeze(1), dim=-1)
        score = F.log_softmax(logits, dim=-1) - dist
        return torch.argmax(score, dim=-1)


# ==========================================
# 5. SIMULATION WRAPPER (FIXED)
# ==========================================

class SimulationWrapper:
    def __init__(self, obstacles, bounds=None):
        """
        Args:
            obstacles: List of (position, radius) tuples
            bounds: Optional dict with 'min' and 'max' keys, each a list of 3 floats [x, y, z]
                   If None, defaults to [-2, -2, -2] to [12, 12, 2]
        """
        self.obstacles = obstacles
        
        # Set default bounds if not provided
        if bounds is None:
            self.bounds = {
                'min': [-2.0, -2.0, -2.0],
                'max': [12.0, 12.0, 2.0]
            }
        else:
            self.bounds = bounds
        
        # Convert to tensors for easier computation
        self.bounds_min = torch.tensor(self.bounds['min'], dtype=torch.float32)
        self.bounds_max = torch.tensor(self.bounds['max'], dtype=torch.float32)
    
    def get_raw_observation(self, current_pos, target_pos, max_obstacles=10):
        """
        Get RAW context observation (for InferenceEngine).
        
        This is the raw, messy context from the world that the InferenceEngine
        will compress into a latent situation. The Actor never sees this directly.
        
        Args:
            current_pos: (1, 3) or (3,) current position tensor
            target_pos: (3,) or (1, 3) target position tensor
            max_obstacles: Maximum number of obstacles to include (default: 10)
                          Observation is always this size, padded with zeros if fewer obstacles
        
        Returns:
            raw_ctx: torch.Tensor of shape (raw_ctx_dim,)
            raw_ctx_dim = 3 (rel_goal) + max_obstacles * 4 (rel_obs_pos + obs_radius)
        """
        # Ensure current_pos is (3,)
        if current_pos.dim() > 1:
            current_pos_flat = current_pos.squeeze(0)
        else:
            current_pos_flat = current_pos
        
        # Ensure target_pos is (3,)
        if target_pos.dim() > 1:
            target_pos_flat = target_pos.squeeze(0)
        else:
            target_pos_flat = target_pos
        
        # Relative goal vector
        rel_goal = target_pos_flat - current_pos_flat  # (3,)
        
        # Get obstacle context: relative positions and radii
        # Sort obstacles by distance to current position (nearest first)
        device = current_pos_flat.device if hasattr(current_pos_flat, 'device') else None
        obstacle_info = []
        for obs_p, obs_r in self.obstacles:
            obs_pos = torch.tensor(obs_p, dtype=torch.float32, device=device)
            rel_obs_pos = obs_pos - current_pos_flat  # (3,)
            obs_radius = torch.tensor([obs_r], dtype=torch.float32, device=device)  # (1,)
            distance = torch.norm(rel_obs_pos)
            obstacle_info.append((distance.item(), rel_obs_pos, obs_radius))
        
        # Sort by distance and take nearest max_obstacles
        obstacle_info.sort(key=lambda x: x[0])
        obstacle_info = obstacle_info[:max_obstacles]
        
        # Build obstacle features: [rel_pos_x, rel_pos_y, rel_pos_z, radius] for each obstacle
        obs_parts = [rel_goal]
        for _, rel_obs_pos, obs_radius in obstacle_info:
            obs_parts.append(rel_obs_pos)  # (3,)
            obs_parts.append(obs_radius)   # (1,)
        
        # Pad with zeros if we have fewer than max_obstacles
        # The actor can learn to ignore zero-padded obstacles (radius=0 means no obstacle)
        # This decouples observation from exact obstacle count - add/remove obstacles freely!
        num_obstacles_present = len(obstacle_info)
        if num_obstacles_present < max_obstacles:
            # Pad with zero vectors: [0, 0, 0, 0] for each missing obstacle
            zeros_3d = torch.zeros(3, dtype=torch.float32, device=device)
            zeros_1d = torch.zeros(1, dtype=torch.float32, device=device)
            for _ in range(max_obstacles - num_obstacles_present):
                obs_parts.append(zeros_3d)  # Zero relative position
                obs_parts.append(zeros_1d)  # Zero radius (indicates no obstacle - actor learns to ignore)
        
        # Concatenate all parts
        raw_ctx = torch.cat(obs_parts, dim=0)  # (3 + max_obstacles*4,)
        
        return raw_ctx
    
    def execute(self, mu_t, sigma_t, current_pos):
        """
        Execute a tube in the world, checking for obstacles and binding violations.
        
        PHILOSOPHY: All physics (obstacles, boundaries, etc.) creates deviation from
        the planned tube. This deviation surfaces through binding failure:
        - Exception: binding failed (deviation > sigma)
        - Magnitude: deviation distance
        - Position: where the violation occurred (actual_p after physics)
        
        Args:
            mu_t: (T, 3) relative tube trajectory
            sigma_t: (T, 1) tube radii
            current_pos: (1, 3) current position
        
        Returns:
            actual_p: (T, 3) actual path taken
        """
        # Ensure mu_t and sigma_t are properly shaped - should be (T, 3) and (T, 1) or (T,)
        if mu_t.dim() > 2:
            mu_t = mu_t.view(-1, 3)
        elif mu_t.dim() == 1:
            mu_t = mu_t.unsqueeze(0)
        
        if sigma_t.dim() > 2:
            sigma_t = sigma_t.view(-1)
        elif sigma_t.dim() == 2:
            sigma_t = sigma_t.squeeze(-1) if sigma_t.shape[1] == 1 else sigma_t[:, 0]
        elif sigma_t.dim() == 0:
            sigma_t = sigma_t.unsqueeze(0)
        
        # Transform relative tube to global coordinates
        mu_global = mu_t + current_pos
        
        actual_path = []
        actual_path.append(current_pos.squeeze().clone())
        
        # If tube is too short (only 1 point), return current position
        if len(mu_global) <= 1:
            return torch.stack(actual_path)
        
        # Execute step by step
        for t in range(1, len(mu_global)):
            expected_p = mu_global[t]
            
            # Handle sigma_t indexing - ensure we get a scalar
            t_idx = min(t, len(sigma_t) - 1)
            affordance_r = sigma_t[t_idx].item() if hasattr(sigma_t[t_idx], 'item') else float(sigma_t[t_idx])
            
            # Start with expected position
            actual_p = expected_p.clone()
            
            # Check for obstacles (physics that creates deviation)
            for obs_p, obs_r in self.obstacles:
                obs_pos = torch.tensor(obs_p, dtype=torch.float32, device=actual_p.device)
                d = torch.norm(actual_p - obs_pos)
                if d < obs_r:
                    # Collision: push away from obstacle (creates deviation from expected)
                    vec = actual_p - obs_pos
                    if d > 1e-6:
                        actual_p = obs_pos + (vec / d) * obs_r
                    else:
                        # If exactly on obstacle, push in a random direction
                        actual_p = obs_pos + torch.tensor([obs_r, 0, 0], dtype=torch.float32, device=actual_p.device)
            
            # Check boundaries (physics that creates deviation - same as obstacles)
            # Boundaries should push back, creating deviation that can trigger binding failure
            device = actual_p.device
            bounds_min = self.bounds_min.to(device)
            bounds_max = self.bounds_max.to(device)
            
            # Check each dimension for boundary violations
            for dim in range(3):
                if actual_p[dim] < bounds_min[dim]:
                    # Violated lower bound: push to boundary (creates deviation)
                    actual_p[dim] = bounds_min[dim]
                elif actual_p[dim] > bounds_max[dim]:
                    # Violated upper bound: push to boundary (creates deviation)
                    actual_p[dim] = bounds_max[dim]
            
            # Binding check: if deviation exceeds sigma, stop early
            # This captures ALL physics deviations: obstacles, boundaries, etc.
            # Exception: binding failed
            # Magnitude: deviation distance
            # Position: where the violation occurred (actual_p)
            deviation = torch.norm(actual_p - expected_p)
            if deviation > affordance_r:
                # Binding broken - physics (obstacle/boundary) created too much deviation
                # Record the exception: position where binding failed
                actual_path.append(actual_p.clone())
                break
            
            actual_path.append(actual_p.clone())
        
        return torch.stack(actual_path) 
        
    def run_episode(self, inference_engine, actor, tkn_processor, start_pos, target_pos, max_observed_obstacles=10, latent_dim=64):
        """
        Run a full episode using the HybridInferenceEngine and Actor for evaluation.
        
        Args:
            inference_engine: The HybridInferenceEngine model to use
            actor: The Actor model to use
            tkn_processor: The TknProcessor for tokenizing observations
            start_pos: Starting position [x, y, z]
            target_pos: Target position [x, y, z]
            max_observed_obstacles: Maximum obstacles in observation
            latent_dim: Dimension of latent situation space
        
        Returns:
            path: numpy array of actual path taken
        """
        current_pos = torch.tensor(start_pos, dtype=torch.float32).view(1, 3)
        target = torch.tensor(target_pos, dtype=torch.float32).view(1, 3)
        
        path_history = [current_pos.squeeze().numpy()]
        steps = 0
        max_steps = 40
        
        # Reset InferenceEngine hidden state at start of episode
        h_state = torch.zeros(1, latent_dim)
        
        # Reset tkn processor
        tkn_processor.reset_episode()
        
        while steps < max_steps:
            rel_goal = target - current_pos
            
            # Get raw observation
            raw_obs = self.get_raw_observation(current_pos, target, max_obstacles=max_observed_obstacles)
            
            with torch.no_grad():
                # Process through tkn
                tkn_output = tkn_processor.process_observation(
                    current_pos, raw_obs, self.obstacles
                )
                lattice_tokens = tkn_output["lattice_tokens"].unsqueeze(0)  # (1, num_heads)
                lattice_traits = tkn_output["lattice_traits"].unsqueeze(0)  # (1, num_heads, 2)
                rel_goal_tensor = tkn_output["rel_goal"].unsqueeze(0)  # (1, 3)
                
                # INFER: Convert tokens + traits + intent to latent situation
                x_n = inference_engine(lattice_tokens, lattice_traits, rel_goal_tensor, h_state)  # (1, latent_dim)
                h_state = x_n  # Update hidden state
                
                # ACT: Propose affordances based on latent situation
                logits, mu_t, sigma_t, knot_mask, _ = actor(x_n, rel_goal, previous_velocity=None)
                
                best_idx = actor.select_port(logits, mu_t, rel_goal).item()
                
                # Get the committed tube (T, 3)
                # We add current_pos to transform from relative -> global
                committed_mu = mu_t[0, best_idx] + current_pos
                committed_sigma = sigma_t[0, best_idx]

            pts = len(committed_mu)
            
            if pts < 5:
                break

            # EXECUTE COMMITMENT (Bind -> Contradict)
            # Start from t=1 because t=0 is our current position
            for t in range(1, pts):
                expected_p = committed_mu[t]
                affordance_r = committed_sigma[t]
                
                # Move
                actual_p = expected_p.clone()
                
                # Simple Physics: Check for Obstacles (creates deviation)
                for obs_p, obs_r in self.obstacles:
                    d = torch.norm(actual_p - torch.tensor(obs_p))
                    if d < obs_r:
                        # Simple bounce/block (pushes back, creating deviation)
                        vec = actual_p - torch.tensor(obs_p)
                        actual_p = torch.tensor(obs_p) + (vec / d) * obs_r
                
                # Check boundaries (physics that creates deviation - same as obstacles)
                # Boundaries push back, creating deviation that can trigger binding failure
                # Convert bounds to same format as obstacles (tensors)
                bounds_min_t = torch.tensor(self.bounds['min'], dtype=torch.float32)
                bounds_max_t = torch.tensor(self.bounds['max'], dtype=torch.float32)
                
                # Check each dimension for boundary violations
                for dim in range(3):
                    if actual_p[dim] < bounds_min_t[dim]:
                        # Violated lower bound: push to boundary (creates deviation)
                        actual_p[dim] = bounds_min_t[dim]
                    elif actual_p[dim] > bounds_max_t[dim]:
                        # Violated upper bound: push to boundary (creates deviation)
                        actual_p[dim] = bounds_max_t[dim]
                
                # BINDING CHECK
                # If reality (actual_p) is too far from plan (expected_p)
                # This captures ALL physics deviations: obstacles, boundaries, etc.
                # Exception: binding failed
                # Magnitude: deviation distance
                # Position: where the violation occurred (actual_p)
                deviation = torch.norm(actual_p - expected_p)
                if deviation > affordance_r:
                    # Update state to where we actually ended up
                    # This is the exception position - where binding failed
                    current_pos = actual_p.view(1, 3)
                    path_history.append(current_pos.squeeze().numpy())
                    break  # Break inner loop to Re-plan
                
                # If valid, commit to this step
                current_pos = actual_p.view(1, 3)
                path_history.append(current_pos.squeeze().numpy())
                
                if torch.norm(current_pos - target) < 1.0:
                    return np.array(path_history)

            steps += 1

        return np.array(path_history)

# ==========================================
# 6. TRAINING & LIVE PLOTTING
# ==========================================

class LivePlotter:
    # Consistent aesthetic color palette
    COLORS = {
        'trajectory': '#6B7280',  # Cool grey for actual paths
        'tube_high_agency': '#2563EB',  # Blue (high agency/confident)
        'tube_low_agency': '#CBD5E1',  # Light blue-grey (low agency/uncertain, darker than background)
        'obstacle': '#EF4444',  # Warm red for obstacles (contrasting)
        'obstacle_alpha': 0.25,  # Obstacle transparency
        'goal': '#F59E0B',  # Amber/gold for goals (contrasting, warm)
        'goal_reached': '#FBBF24',  # Lighter gold for reached goals
        'start': '#1F2937',  # Dark grey for start point
        'current_pos': '#3B82F6',  # Blue for current position marker
        'boundary': '#9CA3AF',  # Medium grey for boundary wireframe
    }
    
    # Line widths (thinner for cleaner look)
    LINE_WIDTHS = {
        'tube': 1.5,  # Thinner tube centerlines
        'trajectory': 2.0,  # Slightly thicker for actual paths (more visible)
        'boundary': 0.8,  # Thin boundary lines
    }
    
    def __init__(self, obstacles, target_pos, bounds=None):
        """
        Args:
            obstacles: List of (position, radius) tuples
            target_pos: Initial target position
            bounds: Optional dict with 'min' and 'max' keys for bounding box
        """
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.obstacles = obstacles
        self.target_pos = target_pos
        
        # Set default bounds if not provided
        if bounds is None:
            self.bounds = {
                'min': [-2.0, -2.0, -2.0],
                'max': [12.0, 12.0, 2.0]
            }
        else:
            self.bounds = bounds
        
        # Plot bounding box as wireframe
        self._plot_bounding_box()
        
        # Plot obstacles with consistent color
        for obs_p, obs_r in obstacles:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = obs_p[0] + obs_r * np.outer(np.cos(u), np.sin(v))
            y = obs_p[1] + obs_r * np.outer(np.sin(u), np.sin(v))
            z = obs_p[2] + obs_r * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color=self.COLORS['obstacle'], 
                               alpha=self.COLORS['obstacle_alpha'], edgecolor='none')
        
        # Plot target (store marker for updates)
        self.target_marker = self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                       color=self.COLORS['goal'], s=300, marker='*', 
                       edgecolors='#D97706', linewidths=1.5, label='Target', zorder=10)
        
        # Plot start
        self.ax.scatter(0, 0, 0, color=self.COLORS['start'], s=100, 
                        marker='o', edgecolors='white', linewidths=1, label='Start')
        
        # Storage for plot elements (only current episode)
        self.tube_lines = []  # Store line segments for gradient-colored tubes
        self.path_lines = []
        self.tube_spheres = []  # Store sphere surfaces for cleanup (deprecated)
        self.tube_markers = []  # Store tube start/end markers
        self.path_markers = []  # Store path endpoint markers
        self.current_pos_marker = None
        self.current_episode = -1  # Track current episode to detect transitions
        self.reached_goal_markers = []  # Store markers for goals reached in current episode
        
        # Custom colormap for agency visualization: Blue (high agency) -> Light blue-grey (low agency)
        # Create a custom colormap from blue to light blue-grey
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = [self.COLORS['tube_high_agency'], self.COLORS['tube_low_agency']]
        self.agency_cmap = LinearSegmentedColormap.from_list('agency', colors_list, N=256)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Live Training: Tube Warping Visualization\nTube color = Agency (σ): Blue=High Agency, Light Grey=Low Agency')
        self.ax.legend()
        self.ax.set_xlim(self.bounds['min'][0], self.bounds['max'][0])
        self.ax.set_ylim(self.bounds['min'][1], self.bounds['max'][1])
        self.ax.set_zlim(self.bounds['min'][2], self.bounds['max'][2])
        
        # Add colorbar for agency visualization (only once, will be updated)
        # Create a scalar mappable for the colormap
        sm = plt.cm.ScalarMappable(cmap=self.agency_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        self.cbar = self.fig.colorbar(sm, ax=self.ax, pad=0.1, shrink=0.6)
        self.cbar.set_label('Normalized σ (Agency)', rotation=270, labelpad=15)
    
    def _plot_bounding_box(self):
        """Plot the bounding box as a wireframe cube."""
        min_b = self.bounds['min']
        max_b = self.bounds['max']
        
        # Define the 8 vertices of the bounding box
        vertices = np.array([
            [min_b[0], min_b[1], min_b[2]],  # 0: min corner
            [max_b[0], min_b[1], min_b[2]],  # 1
            [max_b[0], max_b[1], min_b[2]],  # 2
            [min_b[0], max_b[1], min_b[2]],  # 3
            [min_b[0], min_b[1], max_b[2]],  # 4
            [max_b[0], min_b[1], max_b[2]],  # 5
            [max_b[0], max_b[1], max_b[2]],  # 6
            [min_b[0], max_b[1], max_b[2]],  # 7: max corner
        ])
        
        # Define the 12 edges of the cube
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]
        
        # Plot edges
        for edge in edges:
            points = vertices[edge]
            self.ax.plot3D(*points.T, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Boundary' if edge == edges[0] else '')
        
    def update(self, mu_t, sigma_t, actual_path, current_pos, episode, step):
        """
        Update the plot with new tube and path information.
        
        Args:
            mu_t: (T, 3) planned tube trajectory (relative)
            sigma_t: (T, 1) tube radii
            actual_path: (T, 3) actual path taken
            current_pos: (1, 3) current position
            episode: current episode number
            step: current step number
        """
        # Clear previous episode's visualization when starting a new episode
        if episode != self.current_episode:
            self._clear_episode()
            self.current_episode = episode
        
        # Keep only last few steps for clarity within current episode
        # Optimized: only clean up when we exceed limit (avoid unnecessary work)
        max_history = 10
        if len(self.tube_lines) > max_history:
            # Remove old items efficiently
            to_remove = len(self.tube_lines) - max_history
            for line in self.tube_lines[:to_remove]:
                line.remove()
            self.tube_lines = self.tube_lines[to_remove:]
        
        if len(self.path_lines) > max_history:
            to_remove = len(self.path_lines) - max_history
            for line in self.path_lines[:to_remove]:
                line.remove()
            self.path_lines = self.path_lines[to_remove:]
        
        # Clean up old markers (optimized: batch removal)
        max_markers = max_history * 2
        if len(self.tube_markers) > max_markers:
            to_remove = len(self.tube_markers) - max_markers
            for marker in self.tube_markers[:to_remove]:
                marker.remove()
            self.tube_markers = self.tube_markers[to_remove:]
        
        if len(self.path_markers) > max_markers:
            to_remove = len(self.path_markers) - max_markers
            for marker in self.path_markers[:to_remove]:
                marker.remove()
            self.path_markers = self.path_markers[to_remove:]
        
        # Spheres are disabled, but keep cleanup code for compatibility
        max_spheres = max_history * 5
        if len(self.tube_spheres) > max_spheres:
            to_remove = len(self.tube_spheres) - max_spheres
            for sphere in self.tube_spheres[:to_remove]:
                sphere.remove()
            self.tube_spheres = self.tube_spheres[to_remove:]
        
        # Transform tube to global coordinates
        mu_global = mu_t + current_pos
        
        # Plot planned tube with radius visualization
        mu_np = mu_global.detach().cpu().numpy()
        sigma_np = sigma_t.detach().cpu().numpy()
        
        # Ensure mu_np is 2D (T, 3)
        if mu_np.ndim > 2:
            mu_np = mu_np.reshape(-1, 3)
        elif mu_np.ndim == 1:
            mu_np = mu_np.reshape(1, -1)
        
        # Ensure sigma_np is 1D and matches mu_np length
        if sigma_np.ndim > 1:
            sigma_np = sigma_np.squeeze()
        if sigma_np.ndim == 0:
            sigma_np = np.array([sigma_np.item()])
        
        # Ensure sigma_np matches mu_np length (simple interpolation without scipy)
        if len(sigma_np) != len(mu_np):
            if len(sigma_np) == 1:
                # Single value: replicate
                sigma_np = np.full(len(mu_np), sigma_np[0])
            elif len(sigma_np) < len(mu_np):
                # Upsample: linear interpolation using numpy
                old_indices = np.linspace(0, len(sigma_np) - 1, len(sigma_np))
                new_indices = np.linspace(0, len(sigma_np) - 1, len(mu_np))
                sigma_np = np.interp(new_indices, old_indices, sigma_np)
            else:
                # Downsample: take evenly spaced samples
                indices = np.linspace(0, len(sigma_np) - 1, len(mu_np)).astype(int)
                sigma_np = sigma_np[indices]
        
        # Normalize sigma values for colormap (0 = low uncertainty, 1 = high uncertainty)
        # Use a fixed range for consistent visualization across moves
        # Typical sigma range: 0.1 to 2.0, but we'll use dynamic normalization per tube
        sigma_min, sigma_max = sigma_np.min(), sigma_np.max()
        if sigma_max > sigma_min:
            sigma_normalized = (sigma_np - sigma_min) / (sigma_max - sigma_min)
        else:
            # All values are the same - use middle of colormap
            sigma_normalized = np.full_like(sigma_np, 0.5)
        
        # Plot tube centerline with gradient coloring based on agency (sigma)
        # Low sigma (confident) = darker purple, High sigma (uncertain) = bright yellow
        # Use efficient line collection approach: plot segments with colors
        if len(mu_np) > 1:
            # For efficiency, plot segments in batches to reduce matplotlib overhead
            # Group consecutive segments with similar colors to reduce number of plot calls
            segment_colors = []
            segment_points = []
            
            # Compute segment colors (average of endpoints)
            for i in range(len(mu_np) - 1):
                seg_sigma = (sigma_normalized[i] + sigma_normalized[i + 1]) / 2.0
                color = self.agency_cmap(seg_sigma)
                segment_colors.append(color)
                segment_points.append((mu_np[i], mu_np[i + 1]))
            
            # Plot segments (batch similar colors together for efficiency)
            # For now, plot individually but this is still much faster than spheres
            for i, (start_pt, end_pt) in enumerate(segment_points):
                segment_line, = self.ax.plot(
                    [start_pt[0], end_pt[0]],
                    [start_pt[1], end_pt[1]],
                    [start_pt[2], end_pt[2]],
                    color=segment_colors[i], alpha=0.85, linestyle='--', 
                    linewidth=self.LINE_WIDTHS['tube'],
                    label='Planned Tube (σ gradient)' if len(self.tube_lines) == 0 and i == 0 else ''
                )
                self.tube_lines.append(segment_line)
        else:
            # Single point: just plot as marker
            tube_line, = self.ax.plot(mu_np[:, 0], mu_np[:, 1], mu_np[:, 2], 
                                      color=self.COLORS['tube_high_agency'], alpha=0.7, 
                                      linestyle='--', linewidth=self.LINE_WIDTHS['tube'], 
                                      marker='o', markersize=6,
                                      label='Planned Tube' if len(self.tube_lines) == 0 else '')
            self.tube_lines.append(tube_line)
        
        # Plot starting point of tube (colored by sigma at start)
        start_point = mu_np[0]
        start_sigma_color = self.agency_cmap(sigma_normalized[0] if len(sigma_normalized) > 0 else 0.5)
        start_marker = self.ax.scatter(start_point[0], start_point[1], start_point[2], 
                                      color=start_sigma_color, s=20, alpha=0.85, marker='o',
                                      edgecolors='white', linewidths=0.5)
        self.tube_markers.append(start_marker)
        
        # Plot end point of tube (colored by sigma at end)
        end_point = mu_np[-1]
        end_sigma_color = self.agency_cmap(sigma_normalized[-1] if len(sigma_normalized) > 0 else 0.5)
        end_marker = self.ax.scatter(end_point[0], end_point[1], end_point[2], 
                                     color=end_sigma_color, s=35, alpha=0.9, marker='x', 
                                     linewidths=1.5, edgecolors='white')
        self.tube_markers.append(end_marker)
        
        # DISABLED: Sphere visualization is very expensive (creates many 3D surfaces)
        # The tube centerline and markers provide sufficient visual information
        # If you want to re-enable, reduce to max 1-2 spheres and lower resolution
        # try:
        #     # Only show 1 sphere at midpoint for performance
        #     if len(mu_np) > 2:
        #         mid_idx = len(mu_np) // 2
        #         cx, cy, cz = mu_np[mid_idx]
        #         radius = float(sigma_np[mid_idx] if sigma_np.size > mid_idx else sigma_np[0])
        #         # Use lower resolution (4x4 instead of 8x8)
        #         u = np.linspace(0, 2 * np.pi, 4)
        #         v = np.linspace(0, np.pi, 4)
        #         u_grid, v_grid = np.meshgrid(u, v)
        #         x = cx + radius * np.sin(v_grid) * np.cos(u_grid)
        #         y = cy + radius * np.sin(v_grid) * np.sin(u_grid)
        #         z = cz + radius * np.cos(v_grid)
        #         sphere_surface = self.ax.plot_surface(x, y, z, color='cyan', alpha=0.15, edgecolor='none')
        #         self.tube_spheres.append(sphere_surface)
        # except:
        #     pass
        
        # Plot actual path taken (optimized: only convert if we have a path)
        if actual_path is not None and len(actual_path) > 0:
            # Convert once and reuse
            if isinstance(actual_path, torch.Tensor):
                path_np = actual_path.detach().cpu().numpy()
            else:
                path_np = np.asarray(actual_path)
            
            # Ensure path_np is 2D
            if path_np.ndim == 1:
                path_np = path_np.reshape(1, 3)
            
            # Plot this step's actual path with cool grey color
            if len(path_np) > 1:
                path_line, = self.ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2],
                                         color=self.COLORS['trajectory'], 
                                         linewidth=self.LINE_WIDTHS['trajectory'], 
                                         alpha=0.9, linestyle='-',
                                         label='Actual Path' if len(self.path_lines) == 0 else '')
                self.path_lines.append(path_line)
                
                # Plot path endpoints (reuse path_np array slicing)
                start_pt = path_np[0]
                end_pt = path_np[-1]
                start_marker = self.ax.scatter(start_pt[0], start_pt[1], start_pt[2],
                                              color=self.COLORS['trajectory'], s=25, 
                                              alpha=0.8, marker='s', edgecolors='white', 
                                              linewidths=0.5)
                self.path_markers.append(start_marker)
                
                end_marker = self.ax.scatter(end_pt[0], end_pt[1], end_pt[2],
                                            color=self.COLORS['trajectory'], s=35, 
                                            alpha=0.9, marker='D', edgecolors='white', 
                                            linewidths=0.5)
                self.path_markers.append(end_marker)
            else:
                # Single point path - just plot as a marker
                single_marker = self.ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2],
                                               color=self.COLORS['trajectory'], s=50, 
                                               alpha=0.8, marker='o', edgecolors='white', 
                                               linewidths=0.5)
                self.path_markers.append(single_marker)
        
        # Update current position marker (large, visible)
        if self.current_pos_marker is not None:
            self.current_pos_marker.remove()
        
        # Optimize: convert once and reuse
        if isinstance(current_pos, torch.Tensor):
            pos_np = current_pos.squeeze().detach().cpu().numpy()
        else:
            pos_np = np.asarray(current_pos).squeeze()
        self.current_pos_marker = self.ax.scatter(pos_np[0], pos_np[1], pos_np[2],
                                                  color=self.COLORS['current_pos'], s=150, 
                                                  marker='o', edgecolors='white', 
                                                  linewidths=2, alpha=0.9,
                                                  label='Current Position', zorder=10)
        
        # Update title with episode/step info and key metrics
        avg_sigma = float(sigma_np.mean()) if len(sigma_np) > 0 else 0.0
        goals_reached_count = len(self.reached_goal_markers)
        self.ax.set_title(f'Move {step} | Avg σ: {avg_sigma:.3f} | Goals: {goals_reached_count}')
        
        # Refresh plot (reduced pause for better performance)
        plt.draw()
        plt.pause(0.01)  # Reduced pause time for better performance
    
    def _clear_episode(self):
        """Clear all visualization elements from the previous episode."""
        # Remove all tube lines
        for line in self.tube_lines:
            line.remove()
        self.tube_lines = []
        
        # Remove all path lines
        for line in self.path_lines:
            line.remove()
        self.path_lines = []
        
        # Remove all markers
        for marker in self.tube_markers:
            marker.remove()
        self.tube_markers = []
        
        for marker in self.path_markers:
            marker.remove()
        self.path_markers = []
        
        # Remove all sphere surfaces
        for sphere in self.tube_spheres:
            sphere.remove()
        self.tube_spheres = []
        
        # Remove current position marker
        if self.current_pos_marker is not None:
            self.current_pos_marker.remove()
            self.current_pos_marker = None
        
        # Remove reached goal markers
        for marker in self.reached_goal_markers:
            marker.remove()
        self.reached_goal_markers = []
    
    def mark_goal_reached(self, goal_pos):
        """
        Mark a goal as reached and add it to the visualization.
        
        Args:
            goal_pos: (3,) numpy array or tensor with goal position
        """
        # Convert to numpy if needed
        if isinstance(goal_pos, torch.Tensor):
            goal_np = goal_pos.detach().cpu().numpy() if goal_pos.requires_grad else goal_pos.cpu().numpy()
        else:
            goal_np = np.array(goal_pos)
        
        # Ensure it's 1D
        if goal_np.ndim > 1:
            goal_np = goal_np.squeeze()
        
        # Add a marker for the reached goal (hollow outline only for contrast with active goal)
        reached_marker = self.ax.scatter(
            goal_np[0], goal_np[1], goal_np[2],
            facecolors='none', s=200, marker='*', 
            edgecolors='#D97706', linewidths=0.8, alpha=0.7, 
            label='Reached Goal' if len(self.reached_goal_markers) == 0 else '',
            zorder=9
        )
        self.reached_goal_markers.append(reached_marker)
        self.ax.legend()
    
    def close(self):
        plt.ioff()
        plt.close(self.fig)


def train_actor(inference_engine, actor, total_moves=500, plot_live=True, max_observed_obstacles=10, latent_dim=64):
    """
    Train the InferenceEngine and Actor together in the environment.
    
    This implements dual-optimization: both models learn simultaneously.
    - InferenceEngine learns to represent context such that Actor can successfully bind
    - Actor learns to use that representation to propose tubes
    
    Training runs for a fixed number of moves, continuously generating new goals when reached.
    Visualization shows only the last 10 moves for clarity.
    
    Args:
        inference_engine: HybridInferenceEngine model (GRU-based World Model with tokenized inputs)
        actor: Actor model (operates on latent situations)
        total_moves: Total number of moves to execute (not episodes)
        plot_live: Whether to show live visualization
        max_observed_obstacles: Maximum number of obstacles in observation (fixed size)
        latent_dim: Dimension of latent situation space
    """
    # Combine parameters for joint optimization
    optimizer = optim.Adam(
        list(inference_engine.parameters()) + list(actor.parameters()), 
        lr=1e-3
    )
    # Multiple obstacles creating a more challenging navigation environment
    # You can freely add/remove obstacles here - the observation will pad/truncate automatically
    obstacles = [
        ([5.0, 5.0, 0.0], 2.5),      # Central obstacle
        ([3.0, 8.0, 0.0], 1.5),      # Upper left
        # ([8.0, 3.0, 0.0], 1.5),      # Lower right
        # ([7.0, 7.0, 0.0], 1.8),      # Upper right cluster
        # ([2.0, 2.0, 0.0], 1.2),      # Lower left
        # ([9.0, 6.0, 0.0], 1.3),      # Right side
        # ([4.0, 9.0, 0.0], 1.4),      # Top area
    ]
    
    # Define environment boundaries
    bounds = {
        'min': [-2.0, -2.0, -2.0],
        'max': [12.0, 12.0, 2.0]
    }
    
    sim = SimulationWrapper(obstacles, bounds=bounds)
    target_pos = torch.tensor([10.0, 10.0, 0.0])
    
    # Calculate raw context dimension: 3 (rel_goal) + max_observed_obstacles * 4
    # This is FIXED regardless of actual obstacle count (uses padding/truncation)
    raw_ctx_dim = 3 + max_observed_obstacles * 4
    
    # Create session directory and file
    artifacts_dir = "/Users/zach/Documents/dev/cfd/tam/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific directory for all dump files
    run_dir = os.path.join(artifacts_dir, f"run_{session_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    session_file = os.path.join(run_dir, f"training_session_{session_timestamp}.json")
    goal_stats_file = os.path.join(run_dir, f"goal_stats_{session_timestamp}.jsonl")
    
    # Initialize tkn processor (required for HybridInferenceEngine)
    vocab_size = inference_engine.vocab_size
    tkn_processor = TknProcessor(quantization_bins=11, quant_range=(-2.0, 2.0), vocab_size=vocab_size)
    tkn_log_file = os.path.join(run_dir, f"tkn_patterns_{session_timestamp}.jsonl")
    print(f"TKN enabled: Pattern discovery logging to {tkn_log_file}")
    print(f"  - {len(tkn_processor.head_names)} heads: {', '.join(tkn_processor.head_names)}")
    print(f"  - Vocab size: {tkn_processor.lattice.vocab_size}")
    print(f"  - Using HybridInferenceEngine: tokens + traits + intent")
    
    # Verify models are compatible
    expected_heads = len(tkn_processor.head_names)
    if inference_engine.num_heads != expected_heads:
        raise ValueError(f"InferenceEngine num_heads mismatch: expected {expected_heads}, "
                       f"but model has {inference_engine.num_heads}")
    
    if inference_engine.latent_dim != latent_dim:
        raise ValueError(f"InferenceEngine latent_dim mismatch: expected {latent_dim}, "
                        f"but model has {inference_engine.latent_dim}")
    
    if hasattr(actor, 'encoder'):
        # Check the first layer input size (should be latent_dim + intent_dim)
        first_layer = list(actor.encoder.children())[0]
        if isinstance(first_layer, nn.Linear):
            expected_input = first_layer.in_features - 3  # Subtract intent_dim
            if expected_input != latent_dim:
                raise ValueError(f"Actor latent_dim mismatch: expected {latent_dim}, "
                               f"but actor expects {expected_input}. "
                               f"Initialize Actor with latent_dim={latent_dim}")
    
    # Training data storage
    training_data = {
        "session_timestamp": session_timestamp,
        "total_moves": total_moves,
        "obstacles": obstacles,
        "bounds": bounds,
        "raw_ctx_dim": raw_ctx_dim,
        "latent_dim": latent_dim,
        "max_observed_obstacles": max_observed_obstacles,
        "initial_target": target_pos.tolist(),
        "use_tokenized": True,  # Always using tokenized system
    }
    
    # Initialize live plotter
    plotter = None
    if plot_live:
        plotter = LivePlotter(obstacles, target_pos.numpy(), bounds=bounds)
    
    print(f"Training for {total_moves} moves... (Session: {session_timestamp})")
    print(f"Run directory: {run_dir}")
    print(f"Raw context dimension: {raw_ctx_dim} (3 goal + {max_observed_obstacles} max obstacles * 4)")
    print(f"Latent dimension: {latent_dim}")
    print(f"Actual obstacles in environment: {len(obstacles)}")
    print(f"Visualization will show last 10 moves only")

    # Initialize training state
    current_pos = torch.zeros((1, 3))
    previous_velocity = None  # Track previous path direction for G1 continuity
    move_count = 0  # Total moves executed
    
    # Reset InferenceEngine hidden state at start
    h_state = torch.zeros(1, latent_dim)  # (1, latent_dim) - initial situation
    
    # Reset tkn processor at start
    tkn_processor.reset_episode()
    
    # Goal tracking for logging (use numpy arrays for better performance)
    goal_start_pos = current_pos.clone()  # Position when current goal was set
    goal_start_move = 0  # Move number when current goal was set
    moves_to_current_goal = 0  # Moves taken toward current goal
    segment_lengths = []  # Track segment lengths for current goal (keep as list for extend)
    agency_values = []  # Track sigma (agency) values for current goal (keep as list for extend)
    
    # Performance optimizations:
    # 1. Batch I/O: Buffer tkn logs and write in batches (reduces file I/O overhead)
    # 2. Reduced plot frequency: Update visualization every N moves (reduces matplotlib overhead)
    # 3. Disabled expensive sphere visualization (was creating many 3D surfaces)
    # 4. Optimized tensor operations: Avoid unnecessary copies, use efficient numpy conversions
    # 5. Optimized statistics: Use float32 arrays, compute stats efficiently
    tkn_log_buffer = []  # Buffer tkn logs to write in batches
    tkn_log_batch_size = 10  # Write tkn logs every N moves
    plot_update_frequency = 1  # Update plot every N moves (1 = every move, 2 = every other move, etc.)
    
    # Main training loop: run for total_moves
    while move_count < total_moves:
        # 1. INFER: Convert raw world data to latent situation
        raw_obs = sim.get_raw_observation(current_pos, target_pos, max_obstacles=max_observed_obstacles)  # (raw_ctx_dim,)
        raw_obs = raw_obs.unsqueeze(0)  # (1, raw_ctx_dim) for batch dimension
        
        # Process through tkn (always enabled)
        tkn_output = tkn_processor.process_observation(
            current_pos, raw_obs.squeeze(0), obstacles
        )
        lattice_tokens = tkn_output["lattice_tokens"].unsqueeze(0)  # (1, num_heads)
        surprise_mask = tkn_output["surprise_mask"]  # (num_heads,)
        lattice_traits = tkn_output["lattice_traits"].unsqueeze(0)  # (1, num_heads, 2)
        rel_goal_tensor = tkn_output["rel_goal"].unsqueeze(0)  # (1, 3) high-fidelity intent
        
        # Buffer tkn logs for batch writing (much faster than writing every move)
        log_entry = {
            "move": move_count,
            "lattice_tokens": lattice_tokens.squeeze(0).tolist(),
            "surprise_mask": surprise_mask.tolist(),
            "lattice_traits": lattice_traits.squeeze(0).tolist(),
            "buffer_hashes": tkn_output["buffer_hashes"].tolist(),
            **tkn_output["metadata"]
        }
        tkn_log_buffer.append(log_entry)
        
        # Write in batches to reduce I/O overhead
        if len(tkn_log_buffer) >= tkn_log_batch_size:
            with open(tkn_log_file, 'a') as f:
                for entry in tkn_log_buffer:
                    json.dump(entry, f)
                    f.write('\n')
            tkn_log_buffer.clear()
        
        # Use HybridInferenceEngine: tokens + traits + intent
        x_n = inference_engine(lattice_tokens, lattice_traits, rel_goal_tensor, h_state)  # (1, latent_dim)
        
        h_state = x_n.detach()  # Pass memory forward (detach to prevent backprop through time)
        
        rel_goal = target_pos - current_pos 
        
        # 2. ACT: Propose affordances based on latent situation x_n
        # Actor operates on latent representation, not raw spatial features
        logits, mu_t, sigma_t, knot_mask, knot_steps = actor(x_n, rel_goal, previous_velocity=previous_velocity)
        
        # Selection (Categorical sampling for exploration)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        idx = m.sample()
        
        # Extract selected tube - ensure we get (T, 3) shape, not (1, T, 3)
        idx_int = idx.item() if isinstance(idx, torch.Tensor) else idx
        selected_tube = mu_t[0, idx_int].squeeze(0)  # Remove any extra dimensions
        selected_sigma = sigma_t[0, idx_int].squeeze(0)
        
        # Ensure shapes are correct
        if selected_tube.dim() > 2:
            selected_tube = selected_tube.view(-1, 3)
        if selected_sigma.dim() > 2:
            selected_sigma = selected_sigma.view(-1, 1)
            
        # Track agency (sigma) values for current goal (optimized: single item() call)
        agency_values.append(float(selected_sigma.mean()))
        
        # Track segment lengths (distances between consecutive points in the tube)
        # Only calculate if we have multiple points (optimization)
        if len(selected_tube) > 1:
            # Optimized: compute norm in one go, convert to list efficiently
            tube_segments = selected_tube[1:] - selected_tube[:-1]
            segment_lengths_for_move = torch.norm(tube_segments, dim=-1).detach().cpu()
            segment_lengths.extend(segment_lengths_for_move.tolist())
        
        # Execute the tube
        actual_p = sim.execute(selected_tube, selected_sigma, current_pos)
        
        # 3. SIMPLIFIED PRINCIPLED LOSS: Agency vs. Contradiction
        # With GRU and Residual Knots, we can use a cleaner formulation
        
        # A) CONTRADICTION: Did we stay in the tube?
        # Squared deviation makes hitting obstacles exponentially more painful
        # Optimized: avoid creating full expected_p if not needed
        min_len = min(len(actual_p), len(selected_tube))
        if min_len > 0:
            # Compute deviation efficiently
            expected_p_slice = selected_tube[:min_len] + current_pos
            actual_p_slice = actual_p[:min_len]
            deviation = torch.norm(actual_p_slice - expected_p_slice, dim=-1, keepdim=True)
            binding_loss = torch.mean((deviation**2) / (selected_sigma[:min_len] + 1e-6))
        else:
            binding_loss = torch.tensor(0.0, device=selected_tube.device)
            
        # B) AGENCY COST: How 'expensive' was this path?
        # Extract selected knot_steps for this port
        selected_knot_steps = knot_steps[0, idx_int]  # (K, 3) - residual offsets
        
        # Penalize the sum of squared offsets (shorter, straighter paths are cheaper)
        path_cost = torch.mean(selected_knot_steps**2)
        
        # Penalize uncertainty (narrower tubes are better)
        # MODULATE WITH SURPRISE: Novel patterns increase sigma (expand tube, move cautiously)
        base_uncertainty_cost = torch.mean(selected_sigma**2)
        
        if surprise_mask is not None:
            # Surprise mechanism: Novel patterns -> higher sigma -> more cautious
            surprise_factor = 1.0 + 0.3 * surprise_mask.float().mean().item()  # 1.0 to 1.3 multiplier
            # Note: We don't directly modify sigma here, but we could scale the cost
            uncertainty_cost = base_uncertainty_cost * surprise_factor
        else:
            uncertainty_cost = base_uncertainty_cost
        
        # C) INTENT LOSS: Direct supervision to move toward goal
        # Optimized: avoid unnecessary unsqueeze operations
        tube_endpoint_global = selected_tube[-1] + current_pos.squeeze()
        intent_loss = F.mse_loss(tube_endpoint_global, target_pos)
        
        # Simplified Principled Loss
        loss = (1.0 * binding_loss          # Contradiction: stay in the tube
               + 0.1 * path_cost            # Agency: minimize displacement (residual offsets)
               + 0.05 * uncertainty_cost    # Agency: minimize uncertainty (narrower tubes)
               + 0.5 * intent_loss)         # Intent: move toward goal
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update position and track velocity for G1 continuity
        current_pos = actual_p[-1].view(1, 3).detach()
        move_count += 1
        moves_to_current_goal += 1
        
        # Track velocity: direction of actual path taken (for G1 continuity)
        # Use the last segment of the actual path to capture real movement direction
        if len(actual_p) > 1:
            # Use last segment of actual path (what really happened)
            previous_velocity = (actual_p[-1] - actual_p[-2]).detach()  # (3,)
        elif len(actual_p) == 1 and previous_velocity is None:
            # First step: no previous velocity, use a small default direction
            # This will be updated on next step
            previous_velocity = torch.zeros(3, device=current_pos.device)
        # If path is only one point and we have previous velocity, keep it
            
        # Update live plot (only show last 10 moves)
        # Reduce update frequency for performance (update every N moves)
        if plotter is not None and move_count % plot_update_frequency == 0:
            # Use a fixed episode number (0) so we don't clear on every move
            # The max_history logic in LivePlotter will keep only the last 10 moves
            plotter.update(selected_tube, selected_sigma, actual_p, current_pos, 
                         episode=0, step=move_count)
        
        # Check if goal is reached - if so, log goal statistics and generate new goal
        # Optimized: compute distance once and reuse
        goal_distance = torch.norm(current_pos.squeeze() - target_pos).item()
        if goal_distance < 1.0:
            # Calculate statistics for this goal (optimized: compute all stats in one pass)
            if len(segment_lengths) > 0:
                seg_array = np.asarray(segment_lengths, dtype=np.float32)  # Use float32 for speed
                seg_mean = float(seg_array.mean())
                seg_max = float(seg_array.max())
                seg_min = float(seg_array.min())
                seg_std = float(seg_array.std())
            else:
                seg_mean = seg_max = seg_min = seg_std = 0.0
            
            if len(agency_values) > 0:
                agency_array = np.asarray(agency_values, dtype=np.float32)  # Use float32 for speed
                agency_mean = float(agency_array.mean())
                agency_max = float(agency_array.max())
                agency_min = float(agency_array.min())
                agency_std = float(agency_array.std())
            else:
                agency_mean = agency_max = agency_min = agency_std = 0.0
            
            # Log goal statistics
            goal_stats = {
                "goal_number": len(training_data.get("goals_data", [])) + 1,
                "start_position": goal_start_pos.squeeze().tolist(),
                "goal_position": target_pos.tolist(),
                "moves_taken": moves_to_current_goal,
                "segment_lengths": {
                    "mean": seg_mean,
                    "max": seg_max,
                    "min": seg_min,
                    "std": seg_std,
                    "num_segments": len(segment_lengths)
                },
                "agency": {
                    "mean": agency_mean,
                    "max": agency_max,
                    "min": agency_min,
                    "std": agency_std,
                    "num_samples": len(agency_values)
                }
            }
            
            # Write goal statistics to JSONL file
            with open(goal_stats_file, 'a') as f:
                json.dump(goal_stats, f)
                f.write('\n')
            
            # Store in training data
            if "goals_data" not in training_data:
                training_data["goals_data"] = []
            training_data["goals_data"].append(goal_stats)
            
            # Mark the reached goal in visualization
            if plotter is not None:
                plotter.mark_goal_reached(target_pos)
            
            # Generate a new random goal (avoiding obstacles, current position, and respecting boundaries)
            margin = 0.5
            min_bounds = [b + margin for b in bounds['min']]
            max_bounds = [b - margin for b in bounds['max']]
            
            new_goal = torch.tensor([
                random.uniform(min_bounds[0], max_bounds[0]),
                random.uniform(min_bounds[1], max_bounds[1]),
                random.uniform(min_bounds[2], max_bounds[2])
            ])
            
            # Ensure new goal is not too close to current position or obstacles
            min_dist = 3.0
            max_attempts = 50
            attempts = 0
            while (torch.norm(new_goal - current_pos.squeeze()) < min_dist or
                   any(torch.norm(new_goal - torch.tensor(obs[0])) < obs[1] + 1.0 
                       for obs in obstacles)) and attempts < max_attempts:
                new_goal = torch.tensor([
                    random.uniform(min_bounds[0], max_bounds[0]),
                    random.uniform(min_bounds[1], max_bounds[1]),
                    random.uniform(min_bounds[2], max_bounds[2])
                ])
                attempts += 1
            
            # Reset goal tracking for new goal
            target_pos = new_goal
            goal_start_pos = current_pos.clone()
            goal_start_move = move_count
            moves_to_current_goal = 0
            segment_lengths = []
            agency_values = []
            
            # Update plotter with new goal
            if plotter is not None:
                plotter.target_pos = target_pos.detach().numpy() if isinstance(target_pos, torch.Tensor) else target_pos
                if plotter.target_marker is not None:
                    plotter.target_marker.remove()
                plotter.target_marker = plotter.ax.scatter(
                    target_pos[0].item(), target_pos[1].item(), target_pos[2].item(),
                    color=plotter.COLORS['goal'], s=300, marker='*', 
                    edgecolors='#D97706', linewidths=1.5, label='Target', zorder=10
                )
                plotter.ax.legend()
                plt.draw()
            
            # Print goal reached summary
            print(f"Move {move_count}/{total_moves} | Goal reached! | Moves: {goal_stats['moves_taken']} | "
                  f"Segments: {seg_mean:.3f}±{seg_std:.3f} | Agency: {agency_mean:.3f}±{agency_std:.3f}")
    
    # Flush any remaining buffered tkn logs
    if len(tkn_log_buffer) > 0:
        with open(tkn_log_file, 'a') as f:
            for entry in tkn_log_buffer:
                json.dump(entry, f)
                f.write('\n')
        tkn_log_buffer.clear()
    
    # Save training data to file
    with open(session_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"\nTraining complete. Total moves: {move_count}")
    print(f"Session data saved to: {session_file}")
    print(f"Goal statistics saved to: {goal_stats_file}")
    
    # Save tkn statistics
    tkn_stats_file = os.path.join(run_dir, f"tkn_stats_{session_timestamp}.json")
    tkn_stats = {
        "total_novel_patterns": len(tkn_processor.lattice.novel_patterns),
        "head_names": tkn_processor.head_names,
        "vocab_size": tkn_processor.lattice.vocab_size,
        "novel_patterns_sample": list(tkn_processor.lattice.novel_patterns)[:20]  # Sample of novel patterns
    }
    with open(tkn_stats_file, 'w') as f:
        json.dump(tkn_stats, f, indent=2)
    print(f"TKN statistics saved to: {tkn_stats_file}")
    print(f"Total novel patterns discovered: {tkn_stats['total_novel_patterns']}")
    
    print(f"\nAll run files saved to: {run_dir}")
    print(f"  - Training session: {session_file}")
    print(f"  - Goal statistics: {goal_stats_file}")
    print(f"  - TKN patterns: {tkn_log_file}")
    print(f"  - TKN statistics: {tkn_stats_file}")
    
    if plotter is not None:
        print("Close the plot window to exit.")
        plt.ioff()
    plt.show()

# ==========================================
# 7. MAIN & PLOTTING
# ==========================================

if __name__ == "__main__":
    # System configuration
    MAX_OBSERVED_OBSTACLES = 10  # Maximum obstacles in observation (can add/remove obstacles freely!)
    LATENT_DIM = 128  # Dimension of latent situation space
    
    # Hybrid system: Geometric tokens + Lattice traits + High-fidelity intent
    NUM_HEADS = 7  # delta_x, delta_y, delta_z, proximity, goal_dir_x, goal_dir_y, goal_dir_z
    VOCAB_SIZE = 65536  # Token vocabulary size
    TOKEN_EMBED_DIM = 16  # Embedding dimension per token (reduced since we have traits + intent)
    
    inference_engine = HybridInferenceEngine(
        num_heads=NUM_HEADS,
        vocab_size=VOCAB_SIZE,
        token_embed_dim=TOKEN_EMBED_DIM,
        latent_dim=LATENT_DIM
    )
    print(f"Using Hybrid System: {NUM_HEADS} heads, vocab_size={VOCAB_SIZE}")
    print(f"  - Token embeddings: {TOKEN_EMBED_DIM} dim")
    print(f"  - Lattice traits: hub-ness + surprise")
    print(f"  - High-fidelity intent: raw rel_goal vector")
    
    # Initialize Actor (operates on latent situations)
    actor = Actor(latent_dim=LATENT_DIM, intent_dim=3, n_knots=6, interp_res=40)
    
    # Train both models together with live plotting
    train_actor(
        inference_engine=inference_engine,
        actor=actor,
        total_moves=500,  # Total number of moves (not episodes)
        plot_live=True,
        max_observed_obstacles=MAX_OBSERVED_OBSTACLES,
        latent_dim=LATENT_DIM
    )