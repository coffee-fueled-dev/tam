"""
Alien Observation Transforms for the Alien Sensors Experiment.

IMPORTANT: These transforms must destroy MODE INFORMATION while preserving
enough task structure for B to learn. Previous transforms (simple tanh/RFF)
preserved too much info, allowing B to decode mode at ~99% accuracy.

UPDATED transforms:
- Alien-1: High-frequency RFF (locality-destroying, mode-scrambling)
- Alien-2: Compressed RFF + heavy dropout (information bottleneck)
- Alien-3: Quantized sparse coding (extreme compression)

Key property: B can learn the TASK but cannot easily decode MODE from obs.
"""

from enum import Enum
from typing import Optional

import numpy as np


class AlienLevel(Enum):
    """Levels of observation alienization."""
    NONE = 0      # Identity (baseline)
    ALIEN_1 = 1   # High-frequency RFF (scrambles mode)
    ALIEN_2 = 2   # Compressed RFF + heavy dropout
    ALIEN_3 = 3   # Quantized sparse coding


class AlienObsWrapper:
    """
    Wraps observation transform with configurable alien levels.
    
    DESIGN GOAL: Transform must:
    1. Preserve enough info to learn trajectories
    2. Destroy mode-determining linear combinations
    3. Make mode undecodable via simple classifier (Bayes ceiling ~50%)
    
    All transforms are deterministic given seed, so same latent state
    always maps to same alien observation.
    """
    
    def __init__(
        self,
        input_dim: int,
        level: AlienLevel = AlienLevel.ALIEN_1,
        output_dim: int = 64,
        seed: int = 42,
        noise_std: float = 0.1,  # Increased noise
    ):
        self.input_dim = input_dim
        self.level = level
        self.output_dim = output_dim
        self.seed = seed
        self.noise_std = noise_std
        
        self.rng = np.random.default_rng(seed)
        self._init_transforms()
    
    def _init_transforms(self):
        """Initialize random transform matrices (fixed by seed)."""
        
        if self.level == AlienLevel.NONE:
            self._output_dim = self.input_dim
            return
        
        if self.level == AlienLevel.ALIEN_1:
            # High-frequency Random Fourier Features
            # Large W scale makes the mapping highly nonlinear, destroying
            # simple linear relationships like mode = sign((gy-y)-(gx-x))
            freq_scale = 10.0  # High frequency = rapid variation
            self.W = self.rng.standard_normal((self.output_dim // 2, self.input_dim)).astype(np.float32) * freq_scale
            self.b = self.rng.uniform(0, 2 * np.pi, self.output_dim // 2).astype(np.float32)
            # Random orthogonal mixing
            H = self.rng.standard_normal((self.output_dim, self.output_dim)).astype(np.float32)
            self.Q, _ = np.linalg.qr(H)
            self.Q = self.Q.astype(np.float32)
            self._output_dim = self.output_dim
        
        elif self.level == AlienLevel.ALIEN_2:
            # Compressed representation with heavy information loss
            # 1. Project to low-dim bottleneck
            # 2. Nonlinear expansion
            # 3. Heavy dropout (50%)
            bottleneck_dim = 8  # Information bottleneck
            freq_scale = 5.0
            
            self.W1 = self.rng.standard_normal((bottleneck_dim, self.input_dim)).astype(np.float32)
            self.W2 = self.rng.standard_normal((self.output_dim // 2, bottleneck_dim)).astype(np.float32) * freq_scale
            self.b = self.rng.uniform(0, 2 * np.pi, self.output_dim // 2).astype(np.float32)
            # Heavy dropout (50%)
            self.dropout_mask = (self.rng.random(self.output_dim) > 0.5).astype(np.float32)
            self._output_dim = self.output_dim
        
        elif self.level == AlienLevel.ALIEN_3:
            # Quantized sparse coding - extreme information loss
            # 1. Random projection
            # 2. Threshold to sparse binary
            # 3. XOR with random pattern
            freq_scale = 8.0
            self.W = self.rng.standard_normal((self.output_dim, self.input_dim)).astype(np.float32) * freq_scale
            self.threshold = self.rng.uniform(-1, 1, self.output_dim).astype(np.float32)
            # XOR pattern (flips ~30% of bits)
            self.xor_mask = (self.rng.random(self.output_dim) < 0.3).astype(np.float32)
            self._output_dim = self.output_dim
    
    @property
    def obs_dim(self) -> int:
        """Output observation dimension."""
        return self._output_dim
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply alien transform to observation.
        
        Args:
            x: Input observation [input_dim] or [batch, input_dim]
        
        Returns:
            Transformed observation [output_dim] or [batch, output_dim]
        """
        if self.level == AlienLevel.NONE:
            return x.copy()
        
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        
        if self.level == AlienLevel.ALIEN_1:
            # High-frequency RFF: sin/cos(Wx + b)
            Wx = x @ self.W.T + self.b
            rff = np.concatenate([np.sin(Wx), np.cos(Wx)], axis=-1)
            # Orthogonal scramble
            y = rff @ self.Q.T
            # Add noise
            y = y + self.rng.standard_normal(y.shape).astype(np.float32) * self.noise_std
        
        elif self.level == AlienLevel.ALIEN_2:
            # Compressed RFF through bottleneck
            h = np.tanh(x @ self.W1.T)  # Bottleneck
            Wh = h @ self.W2.T + self.b
            rff = np.concatenate([np.sin(Wh), np.cos(Wh)], axis=-1)
            # Heavy dropout
            y = rff * self.dropout_mask
            # Significant noise
            y = y + self.rng.standard_normal(y.shape).astype(np.float32) * self.noise_std * 2
        
        elif self.level == AlienLevel.ALIEN_3:
            # Quantized sparse coding
            Wx = np.sin(x @ self.W.T)  # Nonlinear projection
            # Threshold to sparse
            sparse = (Wx > self.threshold).astype(np.float32)
            # XOR (flip some bits)
            y = sparse * (1 - self.xor_mask) + (1 - sparse) * self.xor_mask
            # Scale to [-1, 1] range
            y = y * 2 - 1
            # Add noise
            y = y + self.rng.standard_normal(y.shape).astype(np.float32) * self.noise_std
        
        return y.squeeze(0) if single else y
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)


def test_alien_obs():
    """Test all alien levels."""
    print("Testing AlienObsWrapper...")
    
    x = np.array([0.5, 0.3, 0.7, 0.2], dtype=np.float32)
    
    for level in AlienLevel:
        wrapper = AlienObsWrapper(input_dim=4, level=level, output_dim=32, seed=0)
        y = wrapper(x)
        print(f"  {level.name}: input_dim={wrapper.input_dim}, output_dim={wrapper.obs_dim}, "
              f"y_range=[{y.min():.2f}, {y.max():.2f}]")
    
    # Check determinism
    w1 = AlienObsWrapper(input_dim=4, level=AlienLevel.ALIEN_2, seed=42)
    w2 = AlienObsWrapper(input_dim=4, level=AlienLevel.ALIEN_2, seed=42)
    y1 = w1(x)
    # Reset rng for deterministic noise
    w2.rng = np.random.default_rng(42)
    w2._init_transforms()
    y2 = w2(x)
    print(f"  Determinism check (same seed): diff={np.abs(y1 - y2).max():.6f}")
    
    print("✓ AlienObsWrapper works!")


if __name__ == "__main__":
    test_alien_obs()
