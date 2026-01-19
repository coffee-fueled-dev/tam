"""
TAM v3 - Trajectory-Affordance Model

A modular implementation of the Trajectory-Affordance Model with:
- Tokenized spatial pattern recognition
- Transformer-based inference engine (dimension sequences with attention)
- Latent-agnostic actor for affordance tube generation
"""

__version__ = "3.0.0"

# Core components
from v3.actor import Actor
from v3.inference import TransformerInferenceEngine
from v3.geometry import CausalSpline
from v3.tokenizer import UnifiedTknProcessor, TknProcessor, TknHead, MarkovLattice

# Simulation components
from v3.simulation import SimulationWrapper, train_actor

__all__ = [
    'Actor',
    'TransformerInferenceEngine',
    'CausalSpline',
    'UnifiedTknProcessor',
    'TknHead',
    'MarkovLattice',
    'SimulationWrapper',
    'train_actor',
]
