"""
Experiment harness for running TAM experiments with different environments and actors.

The harness provides a unified interface for:
- Running training loops
- Periodic evaluation
- Saving results and visualizations
- Managing experiment configurations
"""

from .experiment_harness import ExperimentHarness, ExperimentConfig
from .runner import run_experiment
from .plots import (
    plot_outcome_vs_sharpness,
    plot_calibration_curve,
    plot_compute_roi,
    plot_commitment_atlas,
    plot_standard_dashboard,
)

__all__ = [
    "ExperimentHarness",
    "ExperimentConfig",
    "run_experiment",
    "plot_outcome_vs_sharpness",
    "plot_calibration_curve",
    "plot_compute_roi",
    "plot_commitment_atlas",
    "plot_standard_dashboard",
]
