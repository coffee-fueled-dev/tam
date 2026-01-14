"""
Universal plotting utilities for TAM evaluation.
"""

from .latent import plot_latent_scatter, plot_pairwise_cosine_histogram
from .geometry import plot_tube_overlay, plot_energy_landscape, plot_trajectory_pca

__all__ = [
    "plot_latent_scatter",
    "plot_pairwise_cosine_histogram",
    "plot_tube_overlay",
    "plot_energy_landscape",
    "plot_trajectory_pca",
]
