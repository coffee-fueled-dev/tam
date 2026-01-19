"""
Simulation module for TAM v3.

Contains:
- SimulationWrapper: Environment physics and observation handling
- train_actor: Training loop
- environment: Deterministic environment generation utilities
"""

from v3.simulation.wrapper import SimulationWrapper
from v3.simulation.train import train_actor
from v3.simulation.environment import generate_obstacles, validate_obstacle_layout

__all__ = [
    'SimulationWrapper', 
    'train_actor', 
    'generate_training_summary',
    'generate_obstacles',
    'validate_obstacle_layout'
]
