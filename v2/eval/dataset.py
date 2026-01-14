"""
Universal episode dataset and buffer utilities.
"""

from typing import Dict, List, Optional, Iterator
import numpy as np
import torch
from collections import deque


class EpisodeBuffer:
    """
    FIFO buffer for storing episodes.
    Label-free, works with any episode format.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, episode: Dict):
        """Add an episode to the buffer."""
        self.buffer.append(episode)
    
    def sample(self, n: int) -> List[Dict]:
        """Sample n random episodes."""
        if len(self.buffer) == 0:
            return []
        indices = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class EpisodeDataset:
    """
    Universal episode dataset for training.
    Works with any environment that provides episodes.
    """
    
    def __init__(
        self,
        buffer: EpisodeBuffer,
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[List[Dict]]:
        """Iterate over batches of episodes."""
        episodes = list(self.buffer.buffer)
        if self.shuffle:
            np.random.shuffle(episodes)
        
        for i in range(0, len(episodes), self.batch_size):
            batch = episodes[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch
    
    def __len__(self):
        """Number of batches."""
        return len(self.buffer) // self.batch_size


class StratifiedEpisodeBuffer:
    """
    Mode-balanced buffer for environments with discrete modes.
    CMG-specific but generalizable to any labeled environment.
    """
    
    def __init__(self, max_size_per_mode: int = 100, n_modes: Optional[int] = None):
        self.max_size_per_mode = max_size_per_mode
        self.n_modes = n_modes
        self.mode_buffers = {}  # mode -> deque
    
    def add(self, episode: Dict, mode: Optional[int] = None):
        """
        Add episode to buffer.
        
        Args:
            episode: Episode dict
            mode: Mode label (if None, tries to extract from episode['k'])
        """
        if mode is None:
            mode = episode.get('k', 0)
            if isinstance(mode, np.ndarray):
                mode = int(mode[-1])
        
        if mode not in self.mode_buffers:
            self.mode_buffers[mode] = deque(maxlen=self.max_size_per_mode)
        
        self.mode_buffers[mode].append(episode)
    
    def sample_balanced(self, n_per_mode: int) -> List[Dict]:
        """
        Sample balanced batch across modes.
        
        Args:
            n_per_mode: Number of episodes per mode
        
        Returns:
            List of episodes
        """
        episodes = []
        for mode, buffer in self.mode_buffers.items():
            n_available = len(buffer)
            n_sample = min(n_per_mode, n_available)
            if n_sample > 0:
                indices = np.random.choice(n_available, size=n_sample, replace=False)
                episodes.extend([buffer[i] for i in indices])
        return episodes
    
    def get_mode(self, mode: int) -> List[Dict]:
        """Get all episodes for a specific mode."""
        return list(self.mode_buffers.get(mode, deque()))
    
    def __len__(self):
        return sum(len(buf) for buf in self.mode_buffers.values())
    
    def mode_counts(self) -> Dict[int, int]:
        """Get count of episodes per mode."""
        return {mode: len(buf) for mode, buf in self.mode_buffers.items()}
