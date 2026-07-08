"""Synthetic stress-test worlds for the atlas benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import torch

Tensor = torch.Tensor


@dataclass
class ObservationEvent:
    """One labeled observation emitted by a benchmark world."""

    observation: Tensor
    regime_name: str
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkWorld:
    """A deterministic observation stream plus family metadata."""

    family: str
    events: List[ObservationEvent]

    def sequence(self) -> list[ObservationEvent]:
        return list(self.events)


def _event(observation: Tensor, regime_name: str, **metadata: float) -> ObservationEvent:
    return ObservationEvent(
        observation=observation.to(dtype=torch.float32),
        regime_name=regime_name,
        metadata=dict(metadata),
    )


def recurrence_world() -> BenchmarkWorld:
    """Well-separated recurring regimes with gaps."""
    events = [
        _event(torch.tensor([1.0, 1.0]), "left", regime_group=0.0),
        _event(torch.tensor([1.1, 0.9]), "left", regime_group=0.0),
        _event(torch.tensor([4.0, 4.0]), "right", regime_group=1.0),
        _event(torch.tensor([4.1, 4.0]), "right", regime_group=1.0),
        _event(torch.tensor([1.05, 1.0]), "left", regime_group=0.0),
        _event(torch.tensor([1.0, 1.1]), "left", regime_group=0.0),
        _event(torch.tensor([7.0, 1.0]), "rare", regime_group=2.0),
        _event(torch.tensor([4.0, 3.9]), "right", regime_group=1.0),
        _event(torch.tensor([7.1, 1.0]), "rare", regime_group=2.0),
    ]
    return BenchmarkWorld(family="recurrence", events=events)


def noisy_world(noise_scale: float = 0.2) -> BenchmarkWorld:
    """Recurring regimes with deterministic perturbations and irrelevant dims."""
    base_left = torch.tensor([1.0, 1.0, 0.0, 0.0])
    base_right = torch.tensor([4.0, 4.0, 0.0, 0.0])
    offsets = [
        torch.tensor([0.0, 0.0, noise_scale, -noise_scale]),
        torch.tensor([noise_scale, -noise_scale, -noise_scale, noise_scale]),
        torch.tensor([-noise_scale, noise_scale, noise_scale, noise_scale]),
    ]
    events: list[ObservationEvent] = []
    for offset in offsets:
        events.append(_event(base_left + offset, "left_noisy", corrupted=1.0, regime_group=0.0))
    for offset in offsets:
        events.append(_event(base_right + offset, "right_noisy", corrupted=1.0, regime_group=1.0))
    for offset in offsets[:2]:
        events.append(_event(base_left + offset * 0.5, "left_noisy", corrupted=1.0, regime_group=0.0))
    return BenchmarkWorld(family="noise", events=events)


def aliasing_world() -> BenchmarkWorld:
    """Different ground-truth regimes with nearly identical observations."""
    events = [
        _event(torch.tensor([2.0, 2.0]), "alias_a", regime_group=0.0),
        _event(torch.tensor([2.0, 2.0]), "alias_b", regime_group=1.0),
        _event(torch.tensor([2.02, 1.98]), "alias_a", regime_group=0.0),
        _event(torch.tensor([1.98, 2.02]), "alias_b", regime_group=1.0),
        _event(torch.tensor([2.01, 2.01]), "alias_a", regime_group=0.0),
        _event(torch.tensor([1.99, 1.99]), "alias_b", regime_group=1.0),
    ]
    return BenchmarkWorld(family="aliasing", events=events)


def drift_world() -> BenchmarkWorld:
    """Gradually moving regimes plus a late novel regime."""
    events: list[ObservationEvent] = []
    for index in range(4):
        events.append(_event(torch.tensor([1.0 + (0.1 * index), 1.0]), "drift_left", phase=float(index), regime_group=0.0))
    for index in range(4):
        events.append(_event(torch.tensor([3.5 + (0.15 * index), 3.5]), "drift_right", phase=float(index), regime_group=1.0))
    for index in range(4):
        events.append(_event(torch.tensor([1.3 + (0.05 * index), 1.0]), "drift_left", phase=float(index + 4), regime_group=0.0))
    events.append(_event(torch.tensor([6.0, 6.0]), "late_novel", phase=99.0, regime_group=2.0))
    events.append(_event(torch.tensor([6.1, 6.0]), "late_novel", phase=100.0, regime_group=2.0))
    return BenchmarkWorld(family="drift", events=events)


def distractor_delay_world() -> BenchmarkWorld:
    """Recurring regimes with long gaps and shifted delayed returns."""
    events: list[ObservationEvent] = []
    events.extend(
        [
            _event(torch.tensor([1.0, 1.0]), "anchor", regime_group=0.0),
            _event(torch.tensor([1.05, 1.0]), "anchor", regime_group=0.0),
            _event(torch.tensor([4.0, 4.0]), "distractor_0", regime_group=1.0),
            _event(torch.tensor([6.0, 3.0]), "distractor_1", regime_group=2.0),
            _event(torch.tensor([8.0, 1.0]), "distractor_2", regime_group=3.0),
            _event(torch.tensor([6.5, 6.5]), "distractor_3", regime_group=4.0),
            _event(torch.tensor([1.38, 1.34]), "anchor", delay_gap=4.0, regime_group=0.0),
            _event(torch.tensor([1.42, 1.36]), "anchor", regime_group=0.0),
        ]
    )
    return BenchmarkWorld(family="distractor_delay", events=events)


def feature_corruption_world() -> BenchmarkWorld:
    """Same regimes with partial feature corruption strong enough to stress strict configs."""
    events = [
        _event(torch.tensor([1.0, 1.0, 0.0, 0.0]), "clean_left", corrupted=0.0, regime_group=0.0),
        _event(torch.tensor([1.05, 0.95, 0.0, 0.0]), "clean_left", corrupted=0.0, regime_group=0.0),
        _event(torch.tensor([1.28, 0.72, 0.0, 0.0]), "clean_left", corrupted=1.0, regime_group=0.0),
        _event(torch.tensor([1.30, 0.70, 0.0, 0.0]), "clean_left", corrupted=1.0, regime_group=0.0),
        _event(torch.tensor([4.0, 4.0, 0.0, 0.0]), "clean_right", corrupted=0.0, regime_group=1.0),
        _event(torch.tensor([4.05, 3.95, 0.0, 0.0]), "clean_right", corrupted=0.0, regime_group=1.0),
        _event(torch.tensor([4.30, 3.72, 0.0, 0.0]), "clean_right", corrupted=1.0, regime_group=1.0),
        _event(torch.tensor([4.32, 3.68, 0.0, 0.0]), "clean_right", corrupted=1.0, regime_group=1.0),
    ]
    return BenchmarkWorld(family="feature_corruption", events=events)


def overlap_boundary_world() -> BenchmarkWorld:
    """Heavier overlap near the decision boundary between nearby regimes."""
    events = [
        _event(torch.tensor([1.00, 1.00]), "near_a", boundary=0.0, regime_group=0.0),
        _event(torch.tensor([1.08, 1.04]), "near_a", boundary=0.0, regime_group=0.0),
        _event(torch.tensor([1.24, 1.16]), "near_b", boundary=0.0, regime_group=1.0),
        _event(torch.tensor([1.30, 1.20]), "near_b", boundary=0.0, regime_group=1.0),
        _event(torch.tensor([1.14, 1.08]), "boundary_a", boundary=1.0, regime_group=0.0),
        _event(torch.tensor([1.16, 1.10]), "boundary_mix", boundary=1.0, regime_group=2.0),
        _event(torch.tensor([1.18, 1.12]), "boundary_mix", boundary=1.0, regime_group=2.0),
        _event(torch.tensor([1.20, 1.14]), "boundary_b", boundary=1.0, regime_group=1.0),
        _event(torch.tensor([1.10, 1.06]), "near_a", boundary=0.0, regime_group=0.0),
        _event(torch.tensor([1.28, 1.18]), "near_b", boundary=0.0, regime_group=1.0),
    ]
    return BenchmarkWorld(family="overlap_boundary", events=events)


def many_small_regimes_world(n_regimes: int = 6) -> BenchmarkWorld:
    """Many close regimes with low within-regime variance."""
    events: list[ObservationEvent] = []
    for regime_idx in range(n_regimes):
        x_center = 1.0 + (0.28 * regime_idx)
        y_center = 1.0 + (0.18 * (regime_idx % 2))
        regime_name = f"small_{regime_idx}"
        events.append(_event(torch.tensor([x_center, y_center, 0.0, 0.0]), regime_name, regime_group=float(regime_idx)))
        events.append(_event(torch.tensor([x_center + 0.05, y_center - 0.02, 0.0, 0.0]), regime_name, regime_group=float(regime_idx)))
    return BenchmarkWorld(family="many_small_regimes", events=events)


def boundary_world() -> BenchmarkWorld:
    """Backward-compatible alias for the harsher overlap boundary world."""
    return overlap_boundary_world()


def scale_world(n_regimes: int = 8, dim: int = 8) -> BenchmarkWorld:
    """Higher-dimensional separated regimes kept for compatibility."""
    events: list[ObservationEvent] = []
    eye = torch.eye(dim, dtype=torch.float32)
    for regime_idx in range(n_regimes):
        center = torch.zeros(dim, dtype=torch.float32)
        center[regime_idx % dim] = 2.0 + regime_idx
        center[(regime_idx + 1) % dim] = 1.0
        regime_name = f"scale_{regime_idx}"
        events.append(_event(center, regime_name, regime_index=float(regime_idx), regime_group=float(regime_idx)))
        events.append(_event(center + (0.1 * eye[regime_idx % dim]), regime_name, regime_index=float(regime_idx), regime_group=float(regime_idx)))
    return BenchmarkWorld(family="scale", events=events)


def default_variants() -> list[BenchmarkWorld]:
    """Default stress-test worlds for the atlas benchmark suite."""
    return [
        recurrence_world(),
        noisy_world(),
        aliasing_world(),
        distractor_delay_world(),
        feature_corruption_world(),
        overlap_boundary_world(),
        drift_world(),
        many_small_regimes_world(),
    ]
