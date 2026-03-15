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
from v3.tokenizer import UnifiedTknProcessor, TknHead, MarkovLattice
from v3.train_tam import train_tam_system

# TAM contracts (abstract interfaces)
from v3.environment import Environment
from v3.system import TAMSystem
from v3.system_impl import TAMSystemWrapper

# Recording components
from v3.stats_recorder import StatsRecorder, JSONLStatsRecorder
from v3.environment_recorder import EnvironmentRecorder, JSONLEnvironmentRecorder

__all__ = [
    "Actor",
    "TransformerInferenceEngine",
    "CausalSpline",
    "UnifiedTknProcessor",
    "TknHead",
    "MarkovLattice",
    "Environment",
    "TAMSystem",
    "TAMSystemWrapper",
    "train_tam_system",
    "StatsRecorder",
    "JSONLStatsRecorder",
    "EnvironmentRecorder",
    "JSONLEnvironmentRecorder",
]
