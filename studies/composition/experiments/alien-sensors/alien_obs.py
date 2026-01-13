"""
Alien Observation Transforms for the Alien Sensors Experiment.

Three escalating levels of observation "strangeness":
- Alien-1: Deterministic nonlinear scramble (tanh + permute)
- Alien-2: Random Fourier features + orthogonal mixing
- Alien-3: Quantized binary-ish bitstream

Key property: B can still learn the task, but observations share
no human-aligned structure with A.
"""

from enum import Enum
from typing import Optional

import numpy as np


class AlienLevel(Enum):
    """Levels of observation alienization."""
    NONE = 0      # Identity (baseline)
    ALIEN_1 = 1   # Tanh + permute + noise
    ALIEN_2 = 2   # Random Fourier features
    ALIEN_3 = 3   # Quantized bitstream


class AlienObsWrapper:
    """
    Wraps observation transform with configurable alien levels.
    
    All transforms are deterministic given seed, so same latent state
    always maps to same alien observation.
    """
    
    def __init__(
        self,
        input_dim: int,
        level: AlienLevel = AlienLevel.ALIEN_1,
        output_dim: int = 128,
        seed: int = 42,
        noise_std: float = 0.01,
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
            # Alien-1: tanh(Wx + b) + permute
            self.W = self.rng.standard_normal((self.output_dim, self.input_dim)).astype(np.float32)
            self.b = self.rng.standard_normal(self.output_dim).astype(np.float32) * 0.5
            self.perm = self.rng.permutation(self.output_dim)
            self._output_dim = self.output_dim
        
        elif self.level == AlienLevel.ALIEN_2:
            # Alien-2: Random Fourier features + orthogonal mixing
            # RFF: [sin(Wx), cos(Wx)] with random W
            self.W = self.rng.standard_normal((self.output_dim // 2, self.input_dim)).astype(np.float32)
            # Random orthogonal matrix via QR decomposition
            H = self.rng.standard_normal((self.output_dim, self.output_dim)).astype(np.float32)
            self.Q, _ = np.linalg.qr(H)
            self.Q = self.Q.astype(np.float32)
            # Dropout mask (fixed per wrapper instance)
            self.dropout_mask = (self.rng.random(self.output_dim) > 0.1).astype(np.float32)
            self._output_dim = self.output_dim
        
        elif self.level == AlienLevel.ALIEN_3:
            # Alien-3: Quantized bitstream
            # sign(sin(Wx)) -> binary-ish, embed to float
            self.W = self.rng.standard_normal((self.output_dim, self.input_dim)).astype(np.float32)
            # Channel flip noise mask (fixed)
            self.flip_mask = (self.rng.random(self.output_dim) < 0.05).astype(np.float32)
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
            # tanh(Wx + b) + permute + noise
            y = np.tanh(x @ self.W.T + self.b)
            y = y[:, self.perm]
            y = y + self.rng.standard_normal(y.shape).astype(np.float32) * self.noise_std
        
        elif self.level == AlienLevel.ALIEN_2:
            # Random Fourier features
            Wx = x @ self.W.T
            rff = np.concatenate([np.sin(Wx), np.cos(Wx)], axis=-1)
            # Orthogonal mixing
            y = rff @ self.Q.T
            # Fixed dropout
            y = y * self.dropout_mask
            y = y + self.rng.standard_normal(y.shape).astype(np.float32) * self.noise_std
        
        elif self.level == AlienLevel.ALIEN_3:
            # Quantized bitstream
            Wx = x @ self.W.T
            y = np.sign(np.sin(Wx))
            # Channel flip noise (XOR-like)
            y = y * (1 - 2 * self.flip_mask)  # flip where mask is 1
            # Small noise to break exact discretization
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
