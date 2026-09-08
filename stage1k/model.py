"""Stage 1K overlapping / near-merge mode characterization."""

from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

from stage1f.model import (
    CIRCLE,
    AngularCone,
    Commitment,
    ConeSet,
    Refinement,
    circular_distance,
    excess_measure,
    normalize_angle,
)
from stage1i.model import (
    classify_operation,
    cluster_angles,
    exact_support_coverage,
    rebuild_multi_from_history,
    rebuild_single_from_history,
    support_cone,
)
from stage1j.model import HysteresisLearner

LEARNERS = (
    "hysteresis_gap30",
    "window_single",
    "full_circle",
    "hysteresis_gap15",
    "hysteresis_gap45",
)
PRIMARY_LEARNERS = ("hysteresis_gap30", "window_single", "full_circle")
DIAGNOSTIC_LEARNERS = ("hysteresis_gap15", "hysteresis_gap45")
SITUATION_LABEL = "site"

GAP_BY_POLICY = {
    "hysteresis_gap15": 15.0,
    "hysteresis_gap30": 30.0,
    "hysteresis_gap45": 45.0,
}


def edge_gap(center_a: float, center_b: float, half_width: float = 10.0) -> float:
    """Edge gap between two equal half-width modes on the circle."""
    separation = circular_distance(center_a, center_b)
    return float(separation - 2.0 * half_width)


class OverlapLearner:
    """Stage 1J hysteresis / window-single / full-circle with configurable gap."""

    def __init__(
        self,
        policy: str,
        window_size: int = 48,
        split_gap: float | None = None,
        grace_t: int = 64,
    ) -> None:
        if policy not in LEARNERS:
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.window_size = int(window_size)
        self.grace_t = int(grace_t)
        if split_gap is None:
            split_gap = GAP_BY_POLICY.get(policy, 30.0)
        self.split_gap = float(split_gap)
        self.frozen = False
        self._inner: HysteresisLearner | None = None
        self._history: deque[float] | list[float]
        self._cones: list[AngularCone] = []

        if policy.startswith("hysteresis_"):
            self._inner = HysteresisLearner(
                window_size=self.window_size,
                split_gap=self.split_gap,
                grace_t=self.grace_t,
            )
            self._history = self._inner._history
        elif policy == "window_single":
            self._history = deque(maxlen=self.window_size)
        else:
            self._history = []
            self._cones = [AngularCone(0.0, CIRCLE / 2)]

    @property
    def cone_set(self) -> ConeSet:
        if self.policy == "full_circle":
            return ConeSet((AngularCone(0.0, CIRCLE / 2),))
        if self._inner is not None:
            return self._inner.cone_set
        return ConeSet(tuple(self._cones))

    def predict(self, step: int) -> Commitment:
        return Commitment(self.cone_set, step)

    def freeze(self) -> None:
        self.frozen = True
        if self._inner is not None:
            self._inner.freeze()

    def observe(self, angle: float) -> Refinement:
        angle = normalize_angle(angle)
        before = self.cone_set
        if self.policy == "full_circle":
            return Refinement("noop", before, before, angle)
        if self.frozen:
            return Refinement("noop", before, before, angle)
        if self._inner is not None:
            return self._inner.observe(angle)

        assert isinstance(self._history, deque)
        self._history.append(angle)
        after = rebuild_single_from_history(list(self._history))
        self._cones = list(after.cones)
        assert after.contains(angle)
        return Refinement(classify_operation(before, after), before, after, angle)

    def to_dict(self) -> dict[str, object]:
        if self._inner is not None:
            payload = self._inner.to_dict()
            payload["policy"] = self.policy
            return payload
        return {
            "policy": self.policy,
            "window_size": self.window_size,
            "split_gap": self.split_gap,
            "grace_t": self.grace_t,
            "frozen": self.frozen,
            "cones": self.cone_set.to_dict(),
        }


def make_learner(
    policy: str,
    window_size: int | None = None,
    split_gap: float | None = None,
    grace_t: int | None = None,
) -> OverlapLearner:
    return OverlapLearner(
        policy,
        window_size=window_size if window_size is not None else 48,
        split_gap=split_gap,
        grace_t=grace_t if grace_t is not None else 64,
    )


def scenario_modes(protocol: Mapping[str, object], scenario: str) -> list[dict[str, float]]:
    scenarios = protocol["scenarios"]
    assert isinstance(scenarios, Mapping)
    config = scenarios[scenario]
    assert isinstance(config, Mapping)
    modes = config["modes"]
    assert isinstance(modes, Sequence)
    return [dict(mode) for mode in modes]  # type: ignore[arg-type]


__all__ = [
    "CIRCLE",
    "DIAGNOSTIC_LEARNERS",
    "GAP_BY_POLICY",
    "LEARNERS",
    "PRIMARY_LEARNERS",
    "SITUATION_LABEL",
    "AngularCone",
    "Commitment",
    "ConeSet",
    "HysteresisLearner",
    "OverlapLearner",
    "Refinement",
    "circular_distance",
    "classify_operation",
    "cluster_angles",
    "edge_gap",
    "exact_support_coverage",
    "excess_measure",
    "make_learner",
    "normalize_angle",
    "rebuild_multi_from_history",
    "rebuild_single_from_history",
    "scenario_modes",
    "support_cone",
]
