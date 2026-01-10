"""
Experiment harness for running TAM experiments with different environments and actors.

The harness provides a unified interface for:
- Running training loops
- Periodic evaluation
- Saving results and visualizations
- Managing experiment configurations
- Cross-environment transfer testing
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
from .cross_env_runner import (
    CrossEnvTransferHarness,
    TransferConfig,
    run_transfer_experiment,
    MemoryReuseActor,
    PrototypeReuseActor,
    BehavioralReuseActor,
    RandomZActor,
    ShuffledMemoryActor,
    ConeSummary,
    PerEpisodeRecord,
)
from .functor import (
    FunctorConfig,
    FunctorTrainer,
    ConeSignature,
    LinearFunctor,
    MLPFunctor,
    create_functor,
    get_cone_signature,
    collect_cone_dataset,
    run_functor_experiment,
)
from .intent_functor import (
    IntentFunctorConfig,
    IntentFunctorTrainer,
    TubeSignature,
    PairedSample,
    AffineFunctor,
    MLPFunctor,
    create_intent_functor,
    get_tube_signature,
    collect_paired_dataset,
    run_intent_functor_experiment,
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
    "CrossEnvTransferHarness",
    "TransferConfig",
    "run_transfer_experiment",
    "MemoryReuseActor",
    "PrototypeReuseActor",
    "BehavioralReuseActor",
    "RandomZActor",
    "ShuffledMemoryActor",
    "ConeSummary",
    "PerEpisodeRecord",
    # Functor learning
    "FunctorConfig",
    "FunctorTrainer",
    "ConeSignature",
    "LinearFunctor",
    "MLPFunctor",
    "create_functor",
    "get_cone_signature",
    "collect_cone_dataset",
    "run_functor_experiment",
    # Intent functor (factored z)
    "IntentFunctorConfig",
    "IntentFunctorTrainer",
    "TubeSignature",
    "PairedSample",
    "AffineFunctor",
    "MLPFunctor",
    "create_intent_functor",
    "get_tube_signature",
    "collect_paired_dataset",
    "run_intent_functor_experiment",
]
