"""
Convenience functions for running experiments.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .experiment_harness import Environment, ExperimentConfig, ExperimentHarness

try:
    from actor import Actor
except ImportError:
    try:
        from ..actor import Actor  # type: ignore
    except ImportError:
        Actor = None  # type: ignore


def run_experiment(
    env_factory: Callable[[Dict[str, Any]], Environment],
    config: Optional[ExperimentConfig] = None,
    actor_factory: Optional[Callable[[Dict[str, Any]], Actor]] = None,
    visualizations: Optional[list] = None,
) -> tuple:
    """
    Run a complete experiment.
    
    Args:
        env_factory: Function that creates an environment given kwargs
        config: Experiment configuration (creates default if None)
        actor_factory: Optional function that creates an actor given kwargs
        visualizations: Optional list of visualization functions to call
    
    Returns:
        (actor, env, harness) tuple
    """
    if config is None:
        config = ExperimentConfig()
    
    harness = ExperimentHarness(
        config=config,
        env_factory=env_factory,
        actor_factory=actor_factory,
    )
    
    actor, env = harness.train()
    
    # Save results (this will automatically generate standard plots)
    harness.save_results(visualizations=visualizations)
    
    return actor, env, harness
