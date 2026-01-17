"""
TAM v3 - Trajectory-Affordance Model

A modular implementation of the Trajectory-Affordance Model with:
- Tokenized spatial pattern recognition
- Hybrid inference engine (tokens + traits + intent)
- Latent-agnostic actor for affordance tube generation
"""

__version__ = "3.0.0"

# Core components
from v3.actor import Actor
from v3.inference import HybridInferenceEngine
from v3.geometry import CausalSpline
from v3.tokenizer import TknProcessor, TknHead, InvariantLattice

# Simulation components
from v3.simulation import SimulationWrapper, LivePlotter, train_actor

__all__ = [
    'Actor',
    'HybridInferenceEngine',
    'CausalSpline',
    'TknProcessor',
    'TknHead',
    'InvariantLattice',
    'SimulationWrapper',
    'LivePlotter',
    'train_actor',
]
