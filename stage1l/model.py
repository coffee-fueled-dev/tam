"""Stage 1L continuous edge-gap drift learners."""

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
    exact_support_coverage,
    rebuild_single_from_history,
    support_cone,
)
from stage1j.model import HysteresisLearner
from stage1k.model import edge_gap as edge_gap_between

LEARNERS = (
    "binned_hysteresis",
    "pooled_hysteresis",
    "pooled_single",
    "full_circle",
)
REGIME_BINS = ("separate", "boundary", "overlap")


def center_b_from_edge_gap(edge_gap: float, half_width: float = 10.0) -> float:
    return float(2.0 * half_width + edge_gap)


def modes_for_edge_gap(
    edge_gap: float,
    half_width: float = 10.0,
    center_a: float = 0.0,
) -> list[dict[str, float]]:
    center_b = center_b_from_edge_gap(edge_gap, half_width)
    return [
        {"center": float(center_a), "half_width": float(half_width), "weight": 0.5},
        {"center": float(center_b), "half_width": float(half_width), "weight": 0.5},
    ]


def classify_regime(edge_gap: float, bins: Mapping[str, Mapping[str, float]]) -> str:
    separate = bins["separate"]
    overlap = bins["overlap"]
    if edge_gap >= float(separate["edge_gap_min"]):
        return "separate"
    if edge_gap <= float(overlap["edge_gap_max"]):
        return "overlap"
    return "boundary"


class DriftLearner:
    """Pooled or regime-binned sticky hysteresis under continuous edge-gap drift."""

    def __init__(
        self,
        policy: str,
        window_size: int = 48,
        split_gap: float = 30.0,
        grace_t: int = 64,
        bins: Sequence[str] = REGIME_BINS,
    ) -> None:
        if policy not in LEARNERS:
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.window_size = int(window_size)
        self.split_gap = float(split_gap)
        self.grace_t = int(grace_t)
        self.bins = tuple(bins)
        self.frozen = False
        self._stores: dict[str, HysteresisLearner] = {}
        self._history: deque[float] | None = None
        self._cones: list[AngularCone] = []

        if policy == "binned_hysteresis":
            self._stores = {
                name: HysteresisLearner(
                    window_size=self.window_size,
                    split_gap=self.split_gap,
                    grace_t=self.grace_t,
                )
                for name in self.bins
            }
        elif policy == "pooled_hysteresis":
            store = HysteresisLearner(
                window_size=self.window_size,
                split_gap=self.split_gap,
                grace_t=self.grace_t,
            )
            self._stores = {name: store for name in self.bins}
            self._global = store
        elif policy == "pooled_single":
            self._history = deque(maxlen=self.window_size)
        else:
            self._cones = [AngularCone(0.0, CIRCLE / 2)]

    def _store(self, regime: str) -> HysteresisLearner:
        if regime not in self._stores:
            raise ValueError(f"unknown regime: {regime}")
        return self._stores[regime]

    def cone_set(self, regime: str) -> ConeSet:
        if self.policy == "full_circle":
            return ConeSet((AngularCone(0.0, CIRCLE / 2),))
        if self.policy == "pooled_single":
            return ConeSet(tuple(self._cones))
        return self._store(regime).cone_set

    def predict(self, regime: str, step: int) -> Commitment:
        return Commitment(self.cone_set(regime), step)

    def freeze(self) -> None:
        self.frozen = True
        for store in set(self._stores.values()):
            store.freeze()

    def observe(self, regime: str, angle: float) -> Refinement:
        angle = normalize_angle(angle)
        before = self.cone_set(regime)
        if self.policy == "full_circle":
            return Refinement("noop", before, before, angle)
        if self.frozen:
            return Refinement("noop", before, before, angle)
        if self.policy in ("binned_hysteresis", "pooled_hysteresis"):
            return self._store(regime).observe(angle)

        assert self._history is not None
        self._history.append(angle)
        after = rebuild_single_from_history(list(self._history))
        self._cones = list(after.cones)
        assert after.contains(angle)
        return Refinement(classify_operation(before, after), before, after, angle)

    def to_dict(self) -> dict[str, object]:
        if self.policy == "binned_hysteresis":
            stores = {name: self._stores[name].to_dict() for name in self.bins}
        elif self.policy == "pooled_hysteresis":
            stores = {"global": self._global.to_dict()}
        else:
            stores = {"cones": self.cone_set("overlap").to_dict()}
        return {
            "policy": self.policy,
            "window_size": self.window_size,
            "split_gap": self.split_gap,
            "grace_t": self.grace_t,
            "frozen": self.frozen,
            "stores": stores,
        }


def make_learner(
    policy: str,
    window_size: int | None = None,
    split_gap: float | None = None,
    grace_t: int | None = None,
) -> DriftLearner:
    return DriftLearner(
        policy,
        window_size=window_size if window_size is not None else 48,
        split_gap=split_gap if split_gap is not None else 30.0,
        grace_t=grace_t if grace_t is not None else 64,
    )


__all__ = [
    "CIRCLE",
    "LEARNERS",
    "REGIME_BINS",
    "AngularCone",
    "Commitment",
    "ConeSet",
    "DriftLearner",
    "HysteresisLearner",
    "Refinement",
    "center_b_from_edge_gap",
    "circular_distance",
    "classify_regime",
    "edge_gap_between",
    "exact_support_coverage",
    "excess_measure",
    "make_learner",
    "modes_for_edge_gap",
    "normalize_angle",
    "support_cone",
]
