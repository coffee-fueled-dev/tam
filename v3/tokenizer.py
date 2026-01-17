import torch
import numpy as np
import hashlib
from datetime import datetime
from collections import defaultdict

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
    Dimension-agnostic multi-head processor for quantizing geometric data into higher-level motifs.
    
    Dynamically creates heads based on state_dim:
    - One delta head per dimension (movement direction)
    - One proximity head (distance to nearest obstacle)
    - One goal direction head per dimension
    
    This allows the system to work with any N-dimensional state space.
    """
    def __init__(self, state_dim=3, quantization_bins=11, quant_range=(-2.0, 2.0), vocab_size=65536):
        """
        Args:
            state_dim: Dimension of state space (e.g., 3 for 3D, 6 for 6D robotic arm)
            quantization_bins: Number of quantization bins per head
            quant_range: Default quantization range (min, max)
            vocab_size: Token vocabulary size
        """
        self.state_dim = state_dim
        self.quantization_bins = quantization_bins
        self.quant_range = quant_range
        self.vocab_size = vocab_size
        
        # Dynamically create heads based on state_dim
        self.heads = {}
        
        # Delta heads: one per dimension (movement direction)
        for i in range(state_dim):
            head_name = f"delta_{i}"
            self.heads[head_name] = TknHead(head_name, quantization_bins, quant_range, vocab_size)
        
        # Proximity head: distance to nearest obstacle
        self.heads["proximity"] = TknHead("proximity", quantization_bins, quant_range=(-5.0, 10.0), vocab_size=vocab_size)
        
        # Goal direction heads: one per dimension
        for i in range(state_dim):
            head_name = f"goal_dir_{i}"
            self.heads[head_name] = TknHead(head_name, quantization_bins, quant_range, vocab_size)
        
        self.head_names = list(self.heads.keys())
        self.previous_pos = None
        self.previous_rel_goal = None
        
        # Initialize invariant lattice
        self.lattice = InvariantLattice(self.head_names, vocab_size=vocab_size)
        
    def process_observation(self, current_pos, raw_obs, obstacles):
        """
        Process a raw observation through tkn heads and update invariant lattice.
        
        Args:
            current_pos: Current position (state_dim,) tensor
            raw_obs: Raw observation tensor (raw_ctx_dim,)
            obstacles: List of (position, radius) tuples where position is (state_dim,) array
            
        Returns:
            dict with:
                - lattice_tokens: (num_heads,) tensor of token IDs
                - surprise_mask: (num_heads,) bool tensor indicating novel patterns
                - lattice_traits: (num_heads, 2) tensor of [hub_count, surprise] per head
                - buffer_hashes: (num_heads,) tensor of buffer hash IDs
                - metadata: Dict with quantized values, emissions, etc.
        """
        current_pos_np = current_pos.squeeze().cpu().numpy() if isinstance(current_pos, torch.Tensor) else current_pos
        
        # Ensure current_pos_np is the right shape
        if current_pos_np.ndim == 0:
            current_pos_np = np.array([current_pos_np])
        elif current_pos_np.ndim > 1:
            current_pos_np = current_pos_np.flatten()
        
        # Extract relative goal from raw_obs (first state_dim elements)
        rel_goal = raw_obs[:self.state_dim].cpu().numpy() if isinstance(raw_obs, torch.Tensor) else raw_obs[:self.state_dim]
        
        # Ensure rel_goal is the right shape
        if rel_goal.ndim == 0:
            rel_goal = np.array([rel_goal])
        elif rel_goal.ndim > 1:
            rel_goal = rel_goal.flatten()
        
        # Calculate deltas (change in position)
        if self.previous_pos is not None:
            delta = current_pos_np - self.previous_pos
        else:
            delta = np.zeros(self.state_dim)
        
        # Quantize and process deltas (one head per dimension)
        head_emissions = {}
        quantized_values = {}
        
        for i in range(self.state_dim):
            head_name = f"delta_{i}"
            delta_q = self.heads[head_name].quantize(delta[i])
            delta_pattern, delta_novel = self.heads[head_name].process(delta_q)
            head_emissions[head_name] = (delta_pattern, delta_novel) if delta_pattern is not None else None
            quantized_values[head_name] = int(delta_q)
        
        # Calculate proximity to nearest obstacle
        min_proximity = float('inf')
        if obstacles:
            for obs_p, obs_r in obstacles:
                obs_pos = np.array(obs_p)
                # Ensure obs_pos matches state_dim
                if obs_pos.ndim == 0:
                    obs_pos = np.array([obs_pos])
                elif obs_pos.ndim > 1:
                    obs_pos = obs_pos.flatten()
                # Pad or truncate to match state_dim
                if len(obs_pos) < self.state_dim:
                    obs_pos = np.pad(obs_pos, (0, self.state_dim - len(obs_pos)), mode='constant')
                elif len(obs_pos) > self.state_dim:
                    obs_pos = obs_pos[:self.state_dim]
                
                distance = np.linalg.norm(current_pos_np - obs_pos) - obs_r
                min_proximity = min(min_proximity, distance)
        else:
            min_proximity = 10.0  # No obstacles
        
        # Quantize and process proximity
        proximity_q = self.heads["proximity"].quantize(min_proximity)
        proximity_pattern, proximity_novel = self.heads["proximity"].process(proximity_q)
        head_emissions["proximity"] = (proximity_pattern, proximity_novel) if proximity_pattern is not None else None
        quantized_values["proximity"] = int(proximity_q)
        
        # Quantize and process goal direction (one head per dimension)
        for i in range(self.state_dim):
            head_name = f"goal_dir_{i}"
            goal_dir_q = self.heads[head_name].quantize(rel_goal[i] if i < len(rel_goal) else 0.0)
            goal_dir_pattern, goal_dir_novel = self.heads[head_name].process(goal_dir_q)
            head_emissions[head_name] = (goal_dir_pattern, goal_dir_novel) if goal_dir_pattern is not None else None
            quantized_values[head_name] = int(goal_dir_q)
        
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
            "rel_goal": torch.tensor(rel_goal, dtype=torch.float32),  # (state_dim,) high-fidelity intent
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "current_pos": current_pos_np.tolist(),
                "rel_goal": rel_goal.tolist(),
                "delta": delta.tolist(),
                "min_proximity": float(min_proximity),
                "quantized": quantized_values,
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

