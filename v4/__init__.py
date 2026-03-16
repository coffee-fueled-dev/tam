"""Simple reference implementation of TAM v4."""

from v4.core import (
    BindingOutcome,
    BindingRecord,
    ClaimedRegion,
    ContextAtom,
    ContextEpisode,
    ContextWindow,
    LatentTransition,
    PortFiber,
    PortSelection,
    Situation,
)
from v4.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSummary,
    BenchmarkVariant,
    ScorecardRow,
    run_benchmark_suite,
    score_benchmark_results,
)
from v4.runtime_torch import TorchAgent, initial_history
from v4.training import train_agent
from v4.worlds import PlaneWorld

__all__ = [
    "BindingOutcome",
    "BindingRecord",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSummary",
    "BenchmarkVariant",
    "ScorecardRow",
    "ClaimedRegion",
    "ContextAtom",
    "ContextEpisode",
    "ContextWindow",
    "LatentTransition",
    "PlaneWorld",
    "PortFiber",
    "PortSelection",
    "Situation",
    "TorchAgent",
    "initial_history",
    "run_benchmark_suite",
    "score_benchmark_results",
    "train_agent",
]
