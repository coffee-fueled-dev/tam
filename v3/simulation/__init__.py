"""
Simulation module for TAM v3.

Contains:
- SimulationWrapper: Environment physics and observation handling
- LivePlotter: Real-time visualization
- train_actor: Training loop
- analysis: Training progress analysis and plotting
- environment: Deterministic environment generation utilities
"""

from v3.simulation.wrapper import SimulationWrapper
from v3.simulation.plotter import LivePlotter
from v3.simulation.train import train_actor
from v3.simulation.analysis import plot_training_progress, generate_training_summary
from v3.simulation.environment import generate_obstacles, validate_obstacle_layout

__all__ = [
    'SimulationWrapper', 
    'LivePlotter', 
    'train_actor', 
    'plot_training_progress', 
    'generate_training_summary',
    'generate_obstacles',
    'validate_obstacle_layout'
]
