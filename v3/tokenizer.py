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


class MarkovLattice:
    """
    Geometric tokenizer using Markov pairs (hub-to-hub transitions) for online tokenization.
    
    Token IDs are assigned based on transitions: (previous_token, current_pattern) → token_id
    This creates a sparse graph where hubs (frequently seen patterns) accumulate and
    hub-to-hub relationships become meaningful for geometric reasoning.
    """
    def __init__(self, head_names, vocab_size=65536, hub_threshold=3):
        self.head_names = head_names
        self.vocab_size = vocab_size
        self.hub_threshold = hub_threshold  # Minimum hub_count to be a hub
        
        # Current token per head (persistent state for Markov transitions)
        self.current_lattice = torch.zeros(len(head_names), dtype=torch.long)
        
        # Markov transition graph: (prev_token, pattern_hash) -> token_id
        # This IS the tokenization mechanism
        self.markov_transitions = {}  # Dict[Tuple[int, int], int]
        
        # Hub graph: hub_id -> {neighbors: [hub_ids], in_degree: int, transition_count: int, hub_count: int, importance_score: float}
        self.hub_graph = {}  # Dict[int, Dict]
        
        # Hub metadata: hub_id -> {pattern_hash, position_history, embeddings}
        self.hub_metadata = {}  # Dict[int, Dict]
        
        # Pattern to hub mapping: pattern_hash -> hub_id (if hub_count >= threshold)
        self.pattern_to_hub = {}  # Dict[int, int]
        
        # Track novel patterns for surprise detection
        self.novel_patterns = set()
        
        # Importance score weights (can be tuned)
        self.importance_weights = {
            'in_degree': 0.4,  # How many hubs transition to this one
            'hub_count': 0.4,  # Frequency of pattern
            'transition_freq': 0.2  # How often transitions occur
        }
    
    def tokenize_with_markov(self, head_name, pattern, hub_count, surprise, previous_token=None):
        """
        Assign token ID based on Markov transition.
        
        Args:
            head_name: Name of the head (e.g., "dim_0")
            pattern: Pattern list from TknHead.process()
            hub_count: Hub count (frequency) of current pattern
            surprise: Surprise signal (1.0 if novel, else 0.0)
            previous_token: Previous token ID for this head (None if first)
            
        Returns:
            token_id: Assigned token ID based on Markov transition
            is_novel: Whether this transition is novel
        """
        # Hash pattern to get pattern identifier
        pattern_str = f"{head_name}:" + "-".join(map(str, pattern))
        pattern_hash = int(hashlib.md5(pattern_str.encode()).hexdigest(), 16) % self.vocab_size
        
        # Determine token ID based on Markov transition
        if previous_token is None:
            # First token: use pattern hash directly
            token_id = pattern_hash
        else:
            # Markov transition: (previous_token, pattern_hash) -> token_id
            transition_key = (previous_token, pattern_hash)
            
            if transition_key in self.markov_transitions:
                # Known transition: use existing token ID
                token_id = self.markov_transitions[transition_key]
            else:
                # Novel transition: assign new token ID
                # Use hash of transition to ensure stability
                transition_hash = int(hashlib.md5(str(transition_key).encode()).hexdigest(), 16) % self.vocab_size
                token_id = transition_hash
                self.markov_transitions[transition_key] = token_id
                self.novel_patterns.add(pattern_str)
        
        # Update hub graph if pattern is a hub (hub_count >= threshold)
        if hub_count >= self.hub_threshold:
            hub_id = token_id  # Use token_id as hub_id
            
            # Initialize hub if new
            if hub_id not in self.hub_graph:
                self.hub_graph[hub_id] = {
                    'neighbors': [],
                    'in_degree': 0,  # Count of hubs that transition TO this hub
                    'transition_count': 0,  # Total transitions from this hub
                    'hub_count': hub_count,
                    'importance_score': 0.0  # Computed importance
                }
                self.hub_metadata[hub_id] = {
                    'pattern_hash': pattern_hash,
                    'position_history': [],
                    'last_seen': None
                }
            else:
                # Update hub_count if increased
                self.hub_graph[hub_id]['hub_count'] = max(
                    self.hub_graph[hub_id]['hub_count'], hub_count
                )
            
            # Track transition to hub (if previous_token exists)
            if previous_token is not None:
                # Check if previous_token is also a hub
                prev_hub_id = self.pattern_to_hub.get(previous_token, None)
                if prev_hub_id is not None and prev_hub_id != hub_id:
                    # Add transition: prev_hub -> current_hub
                    if hub_id not in self.hub_graph[prev_hub_id]['neighbors']:
                        self.hub_graph[prev_hub_id]['neighbors'].append(hub_id)
                    self.hub_graph[prev_hub_id]['transition_count'] += 1
                    
                    # Update in-degree of current hub
                    self.hub_graph[hub_id]['in_degree'] += 1
            
            # Update importance score
            self._update_importance_score(hub_id)
            
            # Update pattern_to_hub mapping
            self.pattern_to_hub[token_id] = hub_id
        
        return token_id, surprise > 0.0
    
    def _update_importance_score(self, hub_id):
        """
        Compute simplified importance score for a hub.
        
        Combines:
        - In-degree: How many hubs transition to this one (normalized)
        - Hub count: Frequency of pattern (normalized)
        - Transition frequency: How often transitions occur from this hub (normalized)
        
        This is a simplified version of PageRank that can be computed incrementally.
        """
        hub = self.hub_graph[hub_id]
        
        # Normalize components (use max values across all hubs for normalization)
        max_in_degree = max((h['in_degree'] for h in self.hub_graph.values()), default=1)
        max_hub_count = max((h['hub_count'] for h in self.hub_graph.values()), default=1)
        max_transition_count = max((h['transition_count'] for h in self.hub_graph.values()), default=1)
        
        # Normalized components
        norm_in_degree = hub['in_degree'] / max_in_degree if max_in_degree > 0 else 0.0
        norm_hub_count = hub['hub_count'] / max_hub_count if max_hub_count > 0 else 0.0
        norm_transition_freq = hub['transition_count'] / max_transition_count if max_transition_count > 0 else 0.0
        
        # Weighted combination
        importance = (
            self.importance_weights['in_degree'] * norm_in_degree +
            self.importance_weights['hub_count'] * norm_hub_count +
            self.importance_weights['transition_freq'] * norm_transition_freq
        )
        
        hub['importance_score'] = importance
    
    def get_hub_importance(self, hub_id):
        """Get importance score for a hub (0.0 to 1.0)."""
        return self.hub_graph.get(hub_id, {}).get('importance_score', 0.0)
    
    def update(self, head_emissions, hub_counts=None, surprises=None, current_pos=None):
        """
        Update lattice with pattern emissions using Markov tokenization.
        
        Args:
            head_emissions: dict mapping head_name -> (pattern_list, is_novel) or None
            hub_counts: Optional dict mapping head_name -> hub_count (if None, will try to infer)
            surprises: Optional dict mapping head_name -> surprise (if None, will use is_novel from emissions)
            current_pos: Optional (state_dim,) current position for metadata
            
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
                    # Get previous token for Markov transition
                    previous_token = self.current_lattice[i].item()
                    
                    # Get hub_count and surprise
                    if hub_counts is not None and name in hub_counts:
                        hub_count = hub_counts[name]
                    else:
                        hub_count = 0  # Default if not provided
                    
                    if surprises is not None and name in surprises:
                        surprise = surprises[name]
                    else:
                        surprise = 1.0 if is_novel else 0.0
                    
                    # Tokenize using Markov transition
                    token_id, is_novel_transition = self.tokenize_with_markov(
                        name, pattern, hub_count, surprise, previous_token
                    )
                    
                    self.current_lattice[i] = token_id
                    surprise_mask[i] = is_novel_transition
                    
                    # Update hub metadata with position if provided
                    if current_pos is not None and token_id in self.hub_metadata:
                        current_pos_copy = current_pos.copy() if hasattr(current_pos, 'copy') else np.array(current_pos)
                        self.hub_metadata[token_id]['position_history'].append(current_pos_copy)
                        self.hub_metadata[token_id]['last_seen'] = current_pos_copy
        
        return self.current_lattice.clone(), surprise_mask
    
    def query_structural_analogies(self, query_tokens, query_embedding=None, k=5):
        """
        Find similar geometric patterns when encountering surprises.
        
        Uses hub graph to find structural analogies based on:
        - Similar importance scores (hubs with similar centrality)
        - Similar transition patterns (neighbors)
        - Embedding similarity (if embeddings available)
        
        Returns top-k hubs sorted by combined similarity score.
        """
        # For each hub, compute similarity score
        similarities = []
        query_importance = self.get_hub_importance(query_tokens[0].item()) if query_tokens and len(query_tokens) > 0 else 0.0
        
        for hub_id, hub_data in self.hub_graph.items():
            # Importance similarity (hubs with similar importance are analogous)
            importance_sim = 1.0 - abs(hub_data['importance_score'] - query_importance)
            
            # Transition pattern similarity (overlap in neighbors)
            # (Can be enhanced with embedding similarity if embeddings are stored)
            
            # Combined score (weighted)
            similarity_score = importance_sim  # Can add more components
            
            similarities.append((hub_id, similarity_score, hub_data))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def query_trajectory(self, trajectory_points, current_pos):
        """
        Query obstacles/patterns along a proposed trajectory.
        
        Uses hub metadata (position_history) to identify obstacles.
        Returns risk signals based on:
        - Hub density near trajectory points
        - Hub importance (high-importance hubs are more "obstacle-like")
        - Transition patterns (hubs that frequently cause binding failures)
        
        Args:
            trajectory_points: (T, state_dim) proposed trajectory
            current_pos: (state_dim,) current position
            
        Returns:
            risk_signals: (T,) tensor indicating obstacle risk at each point.
        """
        if len(self.hub_metadata) == 0:
            # No hubs yet, return zero risk
            return torch.zeros(len(trajectory_points), dtype=torch.float32)
        
        risk_signals = []
        
        # Convert to numpy if needed
        if isinstance(trajectory_points, torch.Tensor):
            traj_np = trajectory_points.detach().cpu().numpy()
        else:
            traj_np = np.array(trajectory_points)
        
        if isinstance(current_pos, torch.Tensor):
            current_pos_np = current_pos.detach().cpu().numpy() if current_pos.requires_grad else current_pos.cpu().numpy()
        else:
            current_pos_np = np.array(current_pos)
        
        # Ensure 1D
        if current_pos_np.ndim > 1:
            current_pos_np = current_pos_np.flatten()
        
        for traj_point in traj_np:
            risk = 0.0
            
            # Check each hub's position history
            for hub_id, metadata in self.hub_metadata.items():
                if 'position_history' in metadata and len(metadata['position_history']) > 0:
                    # Find minimum distance to any position in hub's history
                    min_dist = float('inf')
                    for pos in metadata['position_history']:
                        pos_array = np.array(pos)
                        if pos_array.ndim > 1:
                            pos_array = pos_array.flatten()
                        dist = np.linalg.norm(traj_point - pos_array)
                        min_dist = min(min_dist, dist)
                    
                    # Risk increases if:
                    # 1. Close to hub positions (obstacle locations)
                    # 2. Hub has high importance (central/important obstacles)
                    hub_importance = self.get_hub_importance(hub_id)
                    
                    # Inverse distance weighting with importance
                    if min_dist < 1.0:  # Within obstacle radius
                        risk += hub_importance / (min_dist + 0.1)
            
            risk_signals.append(min(risk, 10.0))  # Cap risk at 10.0
        
        return torch.tensor(risk_signals, dtype=torch.float32)
    
    def reset_episode(self):
        """Reset episode state (but keep lattice tokens and hub graph - they persist)."""
        pass  # Lattice tokens and hub graph persist across episodes


class UnifiedTknProcessor:
    """
    Unified tokenizer with shared weights across dimensions.
    
    Uses a single TknHead that processes each dimension sequentially,
    with dimension identity encoded via positional information in the
    transformer. This eliminates the need for per-dimension heads.
    
    For transformer-based architecture: outputs dimension-tokenized sequences.
    """
    def __init__(self, quantization_bins=11, quant_range=(-2.0, 2.0), vocab_size=65536, hub_threshold=3):
        """
        Args:
            quantization_bins: Number of quantization bins
            quant_range: Default quantization range (min, max)
            vocab_size: Token vocabulary size
            hub_threshold: Minimum hub_count to be considered a hub (for MarkovLattice)
        """
        self.quantization_bins = quantization_bins
        self.quant_range = quant_range
        self.vocab_size = vocab_size
        self.hub_threshold = hub_threshold
        
        # Single unified head (shared weights across dimensions)
        self.unified_head = TknHead("unified", quantization_bins, quant_range, vocab_size)
        
        # Proximity head (still separate, as it's not dimension-specific)
        self.proximity_head = TknHead("proximity", quantization_bins, 
                                     quant_range=(-5.0, 10.0), vocab_size=vocab_size)
        
        # Lattice for tracking patterns (will be initialized after first observation)
        # Now uses MarkovLattice for hub-based Markov tokenization
        self.lattice = None
        self.head_names = None  # Will be set after first observation
        
        # State tracking
        self.previous_pos = None
        self.previous_rel_goal = None
        self.state_dim = None  # Inferred from first observation
    
    def _ensure_initialized(self, state_dim):
        """Lazy initialization of lattice based on inferred state_dim."""
        if self.state_dim == state_dim and self.lattice is not None:
            return
        
        self.state_dim = state_dim
        # Head names: one per dimension + proximity + energy
        self.head_names = [f"dim_{i}" for i in range(state_dim)] + ["proximity", "energy"]
        self.lattice = MarkovLattice(self.head_names, vocab_size=self.vocab_size, hub_threshold=self.hub_threshold)
    
    def process_observation(self, current_pos, raw_obs, obstacles, max_observed_obstacles=10):
        """
        Process observation using unified head for all dimensions.
        
        Args:
            current_pos: (state_dim,) current position tensor
            raw_obs: Raw observation tensor (raw_ctx_dim,)
                     Structure: [rel_goal (state_dim), rel_obs_0 (state_dim), radius_0 (1), 
                                rel_obs_1 (state_dim), radius_1 (1), ...]
            obstacles: List of (position, radius) tuples (for proximity calculation)
            max_observed_obstacles: Maximum number of obstacles in observation
            
        Returns:
            dict with:
                - dimension_tokens: List of (B,) token IDs per dimension
                - dimension_traits: List of (B, 2) traits per dimension [hub_count, surprise]
                - proximity_token: (B,) token ID for proximity
                - proximity_traits: (B, 2) traits for proximity
                - rel_goal: (state_dim,) relative goal vector
                - lattice_tokens: (num_heads,) legacy format (for compatibility)
                - surprise_mask: (num_heads,) legacy format
                - metadata: Dict with processing details
        """
        # Infer state_dim from current_pos
        if isinstance(current_pos, torch.Tensor):
            current_pos_np = current_pos.squeeze().cpu().numpy()
        else:
            current_pos_np = np.array(current_pos)
        
        if current_pos_np.ndim == 0:
            current_pos_np = np.array([current_pos_np])
        elif current_pos_np.ndim > 1:
            current_pos_np = current_pos_np.flatten()
        
        inferred_state_dim = len(current_pos_np)
        self._ensure_initialized(inferred_state_dim)
        
        # Extract relative goal
        if isinstance(raw_obs, torch.Tensor):
            raw_obs_np = raw_obs.cpu().numpy()
        else:
            raw_obs_np = np.array(raw_obs)
        
        rel_goal_np = raw_obs_np[:inferred_state_dim]
        if rel_goal_np.ndim == 0:
            rel_goal_np = np.array([rel_goal_np])
        elif rel_goal_np.ndim > 1:
            rel_goal_np = rel_goal_np.flatten()
        
        # Extract obstacle relative positions from raw_obs
        # Structure: [rel_goal (state_dim), rel_obs_0 (state_dim), radius_0 (1), ...]
        obstacle_directions = []  # List of (state_dim,) arrays
        obs_start_idx = inferred_state_dim
        obstacle_size = inferred_state_dim + 1  # rel_pos (state_dim) + radius (1)
        
        for i in range(max_observed_obstacles):
            obs_pos_start = obs_start_idx + i * obstacle_size
            obs_pos_end = obs_pos_start + inferred_state_dim
            obs_radius_idx = obs_pos_end
            
            if obs_radius_idx < len(raw_obs_np):
                rel_obs_pos = raw_obs_np[obs_pos_start:obs_pos_end]
                obs_radius = raw_obs_np[obs_radius_idx]
                
                # Only process if obstacle exists (radius > small threshold to avoid floating point issues)
                if obs_radius > 1e-6:
                    obstacle_directions.append(rel_obs_pos)
        
        # Find nearest obstacle direction (for per-dimension tokenization)
        nearest_obs_dir = None
        if obstacle_directions:
            # Find obstacle with minimum distance (most relevant for avoidance)
            distances = [np.linalg.norm(obs_dir) for obs_dir in obstacle_directions]
            nearest_idx = np.argmin(distances)
            nearest_obs_dir = obstacle_directions[nearest_idx]
        
        # Extract boundary distances from raw_obs
        # Structure: [..., boundaries: dist_to_min_0, dist_to_max_0, dist_to_min_1, dist_to_max_1, ...]
        boundary_start_idx = inferred_state_dim + max_observed_obstacles * obstacle_size
        boundary_distances = None
        if boundary_start_idx < len(raw_obs_np):
            boundary_end_idx = boundary_start_idx + 2 * inferred_state_dim
            if boundary_end_idx <= len(raw_obs_np):
                boundary_distances = raw_obs_np[boundary_start_idx:boundary_end_idx]
                # Reshape to (state_dim, 2) where each row is [dist_to_min, dist_to_max] for that dimension
                boundary_distances = boundary_distances.reshape(inferred_state_dim, 2)
        
        # Extract energy values from raw_obs (last 2 elements: energy_value, energy_normalized)
        energy_value = None
        energy_normalized = None
        energy_start_idx = inferred_state_dim + max_observed_obstacles * obstacle_size + 2 * inferred_state_dim
        if energy_start_idx < len(raw_obs_np) and len(raw_obs_np) >= energy_start_idx + 2:
            energy_value = float(raw_obs_np[energy_start_idx])
            energy_normalized = float(raw_obs_np[energy_start_idx + 1])
        
        # Calculate deltas
        if self.previous_pos is not None:
            delta = current_pos_np - self.previous_pos
        else:
            delta = np.zeros(inferred_state_dim)
        
        # Process each dimension with unified head
        dimension_tokens = []
        dimension_traits = []
        head_emissions = {}  # For MarkovLattice.update()
        quantized_values = {}
        hub_counts_dict = {}  # For passing to lattice.update()
        surprises_dict = {}  # For passing to lattice.update()
        
        for dim_idx in range(inferred_state_dim):
            # Quantize delta for this dimension
            delta_q = self.unified_head.quantize(delta[dim_idx])
            delta_pattern, delta_novel = self.unified_head.process(delta_q)
            
            # Quantize goal direction for this dimension
            goal_q = self.unified_head.quantize(rel_goal_np[dim_idx] if dim_idx < len(rel_goal_np) else 0.0)
            goal_pattern, goal_novel = self.unified_head.process(goal_q)
            
            # NEW: Quantize obstacle direction for this dimension
            # This enables learning associations between obstacle positions and binding failures
            if nearest_obs_dir is not None:
                obs_dir_component = nearest_obs_dir[dim_idx] if dim_idx < len(nearest_obs_dir) else 0.0
            else:
                obs_dir_component = 0.0  # No obstacle in this direction
            
            obs_dir_q = self.unified_head.quantize(obs_dir_component)
            obs_dir_pattern, obs_dir_novel = self.unified_head.process(obs_dir_q)
            
            # NEW: Quantize boundary distances for this dimension
            # Distance to min boundary (negative = outside bounds, positive = inside)
            # Distance to max boundary (negative = outside bounds, positive = inside)
            # Use proximity head quant range since boundaries can be far
            if boundary_distances is not None and dim_idx < boundary_distances.shape[0]:
                dist_to_min = boundary_distances[dim_idx, 0]
                dist_to_max = boundary_distances[dim_idx, 1]
                # Use minimum distance (closest boundary) as the "danger" signal
                min_boundary_dist = min(dist_to_min, dist_to_max)
                # Quantize using proximity head range (can be negative if outside bounds)
                boundary_q = self.proximity_head.quantize(min_boundary_dist)
                boundary_pattern, boundary_novel = self.proximity_head.process(boundary_q)
            else:
                min_boundary_dist = 10.0  # Far from boundaries
                boundary_q = self.proximity_head.quantize(min_boundary_dist)
                boundary_pattern, boundary_novel = self.proximity_head.process(boundary_q)
            
            # Get head state (traits) - use delta pattern state
            buffer_hash, hub_count, surprise = self.unified_head.get_head_state()
            
            # Select primary pattern for tokenization (delta is primary)
            primary_pattern = None
            is_novel_primary = False
            if delta_pattern is not None:
                primary_pattern = delta_pattern
                is_novel_primary = delta_novel
            elif goal_pattern is not None:
                primary_pattern = goal_pattern
                is_novel_primary = goal_novel
            elif obs_dir_pattern is not None:
                primary_pattern = obs_dir_pattern
                is_novel_primary = obs_dir_novel
            else:
                # No emission yet, use quantized value as single-element pattern
                primary_pattern = [delta_q]
                is_novel_primary = False
            
            # Store for lattice.update() - use head name matching lattice head_names
            head_name = f"dim_{dim_idx}"
            head_emissions[head_name] = (primary_pattern, is_novel_primary)
            hub_counts_dict[head_name] = hub_count
            surprises_dict[head_name] = surprise
            
            dimension_traits.append(torch.tensor([[hub_count, surprise]], dtype=torch.float32))  # (1, 2)
            
            # Track emissions for metadata (legacy format)
            if delta_pattern is not None:
                head_emissions[f"{head_name}_delta"] = (delta_pattern, delta_novel)
            if goal_pattern is not None:
                head_emissions[f"{head_name}_goal"] = (goal_pattern, goal_novel)
            if obs_dir_pattern is not None:
                head_emissions[f"{head_name}_obs_dir"] = (obs_dir_pattern, obs_dir_novel)
            if boundary_pattern is not None:
                head_emissions[f"{head_name}_boundary"] = (boundary_pattern, boundary_novel)
            
            quantized_values[f"{head_name}_delta"] = int(delta_q)
            quantized_values[f"{head_name}_goal"] = int(goal_q)
            quantized_values[f"{head_name}_obs_dir"] = int(obs_dir_q)
            quantized_values[f"{head_name}_boundary"] = int(boundary_q)
        
        # Process proximity (separate head)
        min_proximity = float('inf')
        if obstacles:
            for obs_p, obs_r in obstacles:
                obs_pos = np.array(obs_p)
                if obs_pos.ndim == 0:
                    obs_pos = np.array([obs_pos])
                elif obs_pos.ndim > 1:
                    obs_pos = obs_pos.flatten()
                
                if len(obs_pos) < inferred_state_dim:
                    obs_pos = np.pad(obs_pos, (0, inferred_state_dim - len(obs_pos)), mode='constant')
                elif len(obs_pos) > inferred_state_dim:
                    obs_pos = obs_pos[:inferred_state_dim]
                
                distance = np.linalg.norm(current_pos_np - obs_pos) - obs_r
                min_proximity = min(min_proximity, distance)
        else:
            min_proximity = 10.0
        
        proximity_q = self.proximity_head.quantize(min_proximity)
        proximity_pattern, proximity_novel = self.proximity_head.process(proximity_q)
        proximity_buffer_hash, proximity_hub_count, proximity_surprise = self.proximity_head.get_head_state()
        
        # Store proximity for lattice.update()
        head_name = "proximity"
        primary_pattern = proximity_pattern if proximity_pattern is not None else [proximity_q]
        head_emissions[head_name] = (primary_pattern, proximity_novel)
        hub_counts_dict[head_name] = proximity_hub_count
        surprises_dict[head_name] = proximity_surprise
        
        proximity_traits = torch.tensor([[proximity_hub_count, proximity_surprise]], dtype=torch.float32)
        quantized_values["proximity"] = int(proximity_q)
        
        # Process energy (tokenize like spatial dimensions)
        if energy_value is not None and energy_normalized is not None:
            # Quantize energy_value using proximity_head (can be large values)
            energy_value_q = self.proximity_head.quantize(energy_value)
            energy_value_pattern, energy_value_novel = self.proximity_head.process(energy_value_q)
            
            # Quantize energy_normalized using unified_head (0-1 range)
            energy_normalized_q = self.unified_head.quantize(energy_normalized)
            energy_normalized_pattern, energy_normalized_novel = self.unified_head.process(energy_normalized_q)
            
            # Use normalized energy as primary pattern (more informative for learning)
            energy_pattern = energy_normalized_pattern if energy_normalized_pattern is not None else [energy_normalized_q]
            energy_novel = energy_normalized_novel
            
            # Get head state for energy (use unified_head since we're using normalized)
            energy_buffer_hash, energy_hub_count, energy_surprise = self.unified_head.get_head_state()
            
            # Store energy for lattice.update()
            head_name = "energy"
            head_emissions[head_name] = (energy_pattern, energy_novel)
            hub_counts_dict[head_name] = energy_hub_count
            surprises_dict[head_name] = energy_surprise
            
            energy_traits = torch.tensor([[energy_hub_count, energy_surprise]], dtype=torch.float32)
            quantized_values["energy_value"] = int(energy_value_q)
            quantized_values["energy_normalized"] = int(energy_normalized_q)
        else:
            # No energy data - use default values
            energy_pattern = [0]
            energy_novel = False
            head_name = "energy"
            head_emissions[head_name] = (energy_pattern, energy_novel)
            hub_counts_dict[head_name] = 0.0
            surprises_dict[head_name] = 0.0
            energy_traits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
            quantized_values["energy_value"] = 0
            quantized_values["energy_normalized"] = 0
        
        # Update lattice with Markov tokenization and position tracking
        # This will tokenize all heads using Markov transitions
        if self.lattice:
            lattice_tokens, surprise_mask = self.lattice.update(
                head_emissions, 
                hub_counts=hub_counts_dict,
                surprises=surprises_dict,
                current_pos=current_pos_np
            )
        else:
            lattice_tokens = torch.zeros(len(self.head_names), dtype=torch.long)
            surprise_mask = torch.zeros(len(self.head_names), dtype=torch.bool)
        
        # Extract dimension tokens from lattice (excluding proximity and energy)
        dimension_tokens = []
        for i, name in enumerate(self.head_names):
            if name != "proximity" and name != "energy":
                token_id = lattice_tokens[i].item()
                dimension_tokens.append(torch.tensor([token_id], dtype=torch.long))
        
        # Extract proximity token
        proximity_token_idx = self.head_names.index("proximity")
        proximity_token_id = lattice_tokens[proximity_token_idx].item()
        proximity_token = torch.tensor([proximity_token_id], dtype=torch.long)
        
        # Update state
        self.previous_pos = current_pos_np.copy()
        self.previous_rel_goal = rel_goal_np.copy()
        
        # Build output
        return {
            "dimension_tokens": dimension_tokens,  # List of (1,) tensors
            "dimension_traits": dimension_traits,  # List of (1, 2) tensors
            "proximity_token": proximity_token,  # (1,) tensor
            "proximity_traits": proximity_traits,  # (1, 2) tensor
            "rel_goal": torch.tensor(rel_goal_np, dtype=torch.float32),  # (state_dim,)
            # Legacy format for compatibility
            "lattice_tokens": lattice_tokens,  # (num_heads,)
            "surprise_mask": surprise_mask,  # (num_heads,)
            "lattice_traits": torch.cat(dimension_traits + [proximity_traits, energy_traits], dim=0),  # (num_heads, 2)
            "buffer_hashes": torch.cat([
                torch.tensor([self.unified_head.get_head_state()[0]] * inferred_state_dim + 
                           [self.proximity_head.get_head_state()[0]] +
                           [self.unified_head.get_head_state()[0]], dtype=torch.long)
            ], dim=0),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "current_pos": current_pos_np.tolist(),
                "rel_goal": rel_goal_np.tolist(),
                "delta": delta.tolist(),
                "min_proximity": float(min_proximity),
                "nearest_obstacle_dir": nearest_obs_dir.tolist() if nearest_obs_dir is not None else None,
                "num_obstacles_detected": len(obstacle_directions),
                "boundary_distances": boundary_distances.tolist() if boundary_distances is not None else None,
                "quantized": quantized_values,
                "emissions": {k: (v[0] if v is not None else None) 
                             for k, v in head_emissions.items()},
                "has_emissions": any(v is not None for v in head_emissions.values()),
                "has_surprise": surprise_mask.any().item() if isinstance(surprise_mask, torch.Tensor) else any(surprise_mask),
            }
        }
    
    def reset_episode(self):
        """Reset state at start of new episode."""
        self.previous_pos = None
        self.previous_rel_goal = None
        if self.lattice:
            self.lattice.reset_episode()
        # Reset unified head state (create new instance to reset)
        self.unified_head = TknHead("unified", self.quantization_bins, self.quant_range, self.vocab_size)
        self.proximity_head = TknHead("proximity", self.quantization_bins, 
                                     quant_range=(-5.0, 10.0), vocab_size=self.vocab_size)


class TknProcessor:
    """
    DEPRECATED: Use UnifiedTknProcessor for transformer-based architecture.
    
    Legacy per-dimension head processor. Kept for backward compatibility.
    """
    def __init__(self, state_dim=3, quantization_bins=11, quant_range=(-2.0, 2.0), vocab_size=65536):
        import warnings
        warnings.warn("TknProcessor is deprecated. Use UnifiedTknProcessor for transformer architecture.", DeprecationWarning)
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
        
        # Initialize Markov lattice
        self.lattice = MarkovLattice(self.head_names, vocab_size=vocab_size, hub_threshold=3)
        
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
        
        # Collect lattice traits from all heads (epistemic status)
        buffer_hashes = []
        hub_counts_dict = {}
        surprises_dict = {}
        
        for head_name in self.head_names:
            buffer_hash, hub_count, surprise = self.heads[head_name].get_head_state()
            buffer_hashes.append(buffer_hash)
            hub_counts_dict[head_name] = float(hub_count)
            surprises_dict[head_name] = float(surprise)
        
        # Update Markov lattice with hub_counts and surprises
        lattice_tokens, surprise_mask = self.lattice.update(
            head_emissions,
            hub_counts=hub_counts_dict,
            surprises=surprises_dict,
            current_pos=current_pos_np
        )
        
        # Convert hub_counts and surprises to lists for compatibility
        hub_counts = [hub_counts_dict[name] for name in self.head_names]
        surprises = [surprises_dict[name] for name in self.head_names]
        
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

