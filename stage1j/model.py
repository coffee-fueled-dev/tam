"""Stage 1J sticky multi-cone hysteresis under mode churn."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping, Sequence

from stage1f.model import (
    CIRCLE,
    EPS,
    AngularCone,
    Commitment,
    ConeSet,
    GeometricLearner,
    Refinement,
    _canonical_cones,
    circular_distance,
    excess_measure,
    merge_cones,
    minimal_covering_cone,
    normalize_angle,
    oracle_support,
)
from stage1i.model import (
    classify_operation,
    cluster_angles,
    exact_support_coverage,
    rebuild_multi_from_history,
    regime_modes,
    support_cone,
)

LEARNERS = ("window_multi", "hysteresis_multi", "cumulative_multi", "full_circle")
SITUATION_LABEL = "site"


@dataclass
class StickyCone:
    cone: AngularCone
    miss_streak: int = 0


class HysteresisLearner:
    """Sticky multi-cone: immediate add, delayed prune after grace T."""

    def __init__(
        self,
        window_size: int = 64,
        split_gap: float = 30.0,
        grace_t: int = 32,
    ) -> None:
        self.policy = "hysteresis_multi"
        self.window_size = int(window_size)
        self.split_gap = float(split_gap)
        self.grace_t = int(grace_t)
        self.frozen = False
        self._history: Deque[float] = deque(maxlen=self.window_size)
        self._sticky: list[StickyCone] = []

    @property
    def cone_set(self) -> ConeSet:
        return ConeSet(tuple(item.cone for item in self._sticky))

    def predict(self, step: int) -> Commitment:
        return Commitment(self.cone_set, step)

    def freeze(self) -> None:
        self.frozen = True

    def _match_index(self, cluster_cone: AngularCone) -> int | None:
        best_index = None
        best_distance = None
        for index, sticky in enumerate(self._sticky):
            distance = circular_distance(sticky.cone.center, cluster_cone.center)
            if distance <= self.split_gap + EPS:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = index
        return best_index

    def observe(self, angle: float) -> Refinement:
        angle = normalize_angle(angle)
        before = self.cone_set
        if self.frozen:
            return Refinement("noop", before, before, angle)

        self._history.append(angle)
        history = list(self._history)
        clusters = cluster_angles(history, self.split_gap)
        cluster_cones = [
            minimal_covering_cone(cluster) for cluster in clusters if cluster
        ]

        matched_sticky: set[int] = set()
        for cluster_cone in cluster_cones:
            match = self._match_index(cluster_cone)
            if match is None:
                self._sticky.append(StickyCone(cluster_cone, miss_streak=0))
                matched_sticky.add(len(self._sticky) - 1)
                continue
            sticky = self._sticky[match]
            # Grow-or-hold: never shrink sticky geometry from a partial window
            # remnant. Whole-cone removal is the prune path after grace T.
            sticky.cone = merge_cones(sticky.cone, cluster_cone)
            sticky.miss_streak = 0
            matched_sticky.add(match)

        survivors: list[StickyCone] = []
        for index, sticky in enumerate(self._sticky):
            if index in matched_sticky:
                survivors.append(sticky)
                continue
            sticky.miss_streak += 1
            if sticky.miss_streak < self.grace_t:
                survivors.append(sticky)
            # else prune
        self._sticky = survivors

        # Ensure current observation is represented (should already be via history).
        after = self.cone_set
        if not after.contains(angle):
            self._sticky.append(StickyCone(AngularCone(angle, 0.0), miss_streak=0))
            # Re-merge if needed
            cones = _canonical_cones([item.cone for item in self._sticky])
            self._sticky = [StickyCone(cone, miss_streak=0) for cone in cones]
            after = self.cone_set

        # Canonical merge overlapping stickies while preserving max miss streak
        # of merged members when centers coincide after merge.
        if len(self._sticky) > 1:
            merged_cones = _canonical_cones([item.cone for item in self._sticky])
            new_sticky: list[StickyCone] = []
            for cone in merged_cones:
                streaks = [
                    item.miss_streak
                    for item in self._sticky
                    if circular_distance(item.cone.center, cone.center)
                    <= self.split_gap + EPS
                ]
                new_sticky.append(
                    StickyCone(cone, miss_streak=max(streaks) if streaks else 0)
                )
            self._sticky = new_sticky
            after = self.cone_set

        assert after.contains(angle)
        operation = classify_operation(before, after)
        return Refinement(operation, before, after, angle)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "window_size": self.window_size,
            "split_gap": self.split_gap,
            "grace_t": self.grace_t,
            "frozen": self.frozen,
            "history": list(self._history),
            "sticky": [
                {
                    "center": item.cone.center,
                    "half_width": item.cone.half_width,
                    "miss_streak": item.miss_streak,
                }
                for item in self._sticky
            ],
            "cones": self.cone_set.to_dict(),
        }


class ChurnLearner:
    """Wrapper exposing a common API for Stage 1J learners."""

    def __init__(
        self,
        policy: str,
        window_size: int | None = 64,
        split_gap: float = 30.0,
        grace_t: int = 32,
    ) -> None:
        if policy not in LEARNERS:
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.window_size = window_size
        self.split_gap = float(split_gap)
        self.grace_t = int(grace_t)
        self.frozen = False
        self._inner: HysteresisLearner | GeometricLearner | None
        self._history: Deque[float] | list[float]
        self._cones: list[AngularCone] = []

        if policy == "hysteresis_multi":
            self._inner = HysteresisLearner(
                window_size=window_size or 64,
                split_gap=split_gap,
                grace_t=grace_t,
            )
            self._history = self._inner._history
        elif policy == "window_multi":
            self._inner = None
            self._history = deque(maxlen=int(window_size or 64))
        elif policy == "cumulative_multi":
            self._inner = GeometricLearner("multi_cone", split_gap=split_gap)
            self._history = []
        else:
            self._inner = None
            self._history = []
            self._cones = [AngularCone(0.0, CIRCLE / 2)]

    @property
    def cone_set(self) -> ConeSet:
        if self.policy == "full_circle":
            return ConeSet((AngularCone(0.0, CIRCLE / 2),))
        if self.policy == "hysteresis_multi":
            assert isinstance(self._inner, HysteresisLearner)
            return self._inner.cone_set
        if self.policy == "cumulative_multi":
            assert isinstance(self._inner, GeometricLearner)
            return self._inner.cone_set
        return ConeSet(tuple(self._cones))

    def predict(self, step: int) -> Commitment:
        return Commitment(self.cone_set, step)

    def freeze(self) -> None:
        self.frozen = True
        if isinstance(self._inner, HysteresisLearner):
            self._inner.freeze()

    def observe(self, angle: float) -> Refinement:
        angle = normalize_angle(angle)
        before = self.cone_set
        if self.policy == "full_circle":
            return Refinement("noop", before, before, angle)
        if self.frozen:
            return Refinement("noop", before, before, angle)
        if self.policy == "hysteresis_multi":
            assert isinstance(self._inner, HysteresisLearner)
            return self._inner.observe(angle)
        if self.policy == "cumulative_multi":
            assert isinstance(self._inner, GeometricLearner)
            return self._inner.observe(angle)

        # window_multi
        assert isinstance(self._history, deque)
        self._history.append(angle)
        after = rebuild_multi_from_history(list(self._history), self.split_gap)
        self._cones = list(after.cones)
        assert after.contains(angle)
        return Refinement(classify_operation(before, after), before, after, angle)

    def to_dict(self) -> dict[str, object]:
        if self.policy == "hysteresis_multi":
            assert isinstance(self._inner, HysteresisLearner)
            return self._inner.to_dict()
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
    split_gap: float = 30.0,
    grace_t: int = 32,
) -> ChurnLearner:
    return ChurnLearner(
        policy,
        window_size=window_size,
        split_gap=split_gap,
        grace_t=grace_t,
    )


def mode_b_coverage(cone_set: ConeSet, protocol: Mapping[str, object]) -> float:
    modes = protocol["modes"]
    assert isinstance(modes, Mapping)
    return exact_support_coverage(cone_set, [dict(modes["B"])])


__all__ = [
    "CIRCLE",
    "LEARNERS",
    "SITUATION_LABEL",
    "AngularCone",
    "ChurnLearner",
    "Commitment",
    "ConeSet",
    "HysteresisLearner",
    "Refinement",
    "classify_operation",
    "exact_support_coverage",
    "excess_measure",
    "make_learner",
    "mode_b_coverage",
    "normalize_angle",
    "regime_modes",
    "support_cone",
]
