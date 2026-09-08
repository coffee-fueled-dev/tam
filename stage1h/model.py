"""Stage 1H rolling-window geometric cone learners with narrowing."""

from __future__ import annotations

from collections import deque
from typing import Deque, Mapping, Sequence

from stage1f.model import (
    CIRCLE,
    EPS,
    AngularCone,
    Commitment,
    ConeSet,
    Refinement,
    excess_measure,
    minimal_covering_cone,
    normalize_angle,
    oracle_support,
)

LEARNERS = ("window64", "cumulative", "frozen_a1", "full_circle")
SITUATION_LABEL = "site"


def cone_contains_cone(outer: ConeSet, inner: ConeSet) -> bool:
    if not inner.cones:
        return True
    if not outer.cones:
        return False
    for cone in inner.cones:
        samples = (
            cone.center,
            normalize_angle(cone.center - cone.half_width),
            normalize_angle(cone.center + cone.half_width),
        )
        if not all(outer.contains(sample) for sample in samples):
            return False
    return True


def classify_operation(before: ConeSet, after: ConeSet) -> str:
    if abs(before.measure - after.measure) < EPS and cone_contains_cone(before, after) and cone_contains_cone(after, before):
        return "noop"
    if before.measure < EPS and after.measure >= EPS:
        return "add"
    before_in_after = cone_contains_cone(after, before)
    after_in_before = cone_contains_cone(before, after)
    if before_in_after and after.measure > before.measure + EPS:
        return "widen"
    if after_in_before and after.measure + EPS < before.measure:
        return "narrow"
    if abs(before.measure - after.measure) < EPS and not (
        before_in_after and after_in_before
    ):
        return "shift"
    if after.measure > before.measure + EPS:
        return "widen"
    if after.measure + EPS < before.measure:
        return "narrow"
    return "shift"


def rebuild_from_history(history: Sequence[float]) -> ConeSet:
    if not history:
        return ConeSet(())
    return ConeSet((minimal_covering_cone(history),))


class NarrowingLearner:
    """Geometric cone learner with optional rolling-window reconstruction."""

    def __init__(self, policy: str, window_size: int | None = 64) -> None:
        if policy not in LEARNERS and not policy.startswith("window"):
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.window_size = window_size
        self.frozen = False
        self._history: Deque[float] | list[float]
        if policy.startswith("window") and window_size is not None:
            self._history = deque(maxlen=int(window_size))
        else:
            self._history = []
        self._cones: list[AngularCone] = []
        if policy == "full_circle":
            self._cones = [AngularCone(0.0, CIRCLE / 2)]

    @property
    def cone_set(self) -> ConeSet:
        if self.policy == "full_circle":
            return ConeSet((AngularCone(0.0, CIRCLE / 2),))
        return ConeSet(tuple(self._cones))

    def predict(self, step: int) -> Commitment:
        return Commitment(self.cone_set, step)

    def freeze(self) -> None:
        self.frozen = True

    def observe(self, angle: float) -> Refinement:
        angle = normalize_angle(angle)
        before = self.cone_set

        if self.policy == "full_circle":
            after = before
            return Refinement("noop", before, after, angle)

        if self.frozen:
            return Refinement("noop", before, before, angle)

        if self.policy.startswith("window"):
            self._history.append(angle)
            after = rebuild_from_history(list(self._history))
            self._cones = list(after.cones)
            operation = classify_operation(before, after)
            assert after.contains(angle)
            return Refinement(operation, before, after, angle)

        # cumulative / frozen_a1 while unfrozen: grow-only covering cone
        if not self._history and not self._cones:
            self._history.append(angle)
            self._cones = [AngularCone(angle, 0.0)]
            after = self.cone_set
            return Refinement("add", before, after, angle)

        if isinstance(self._history, list):
            self._history.append(angle)
        else:
            self._history.append(angle)

        if before.contains(angle):
            return Refinement("noop", before, before, angle)

        after = rebuild_from_history(list(self._history))
        # Cumulative never shrinks: if rebuild somehow smaller, keep union by
        # covering previous endpoints plus new angle.
        if after.measure + EPS < before.measure:
            endpoints = [angle]
            for cone in before.cones:
                endpoints.extend(
                    [
                        cone.center,
                        normalize_angle(cone.center - cone.half_width),
                        normalize_angle(cone.center + cone.half_width),
                    ]
                )
            after = ConeSet((minimal_covering_cone(endpoints),))
        self._cones = list(after.cones)
        operation = classify_operation(before, after)
        assert after.contains(angle)
        # Cumulative must not narrow.
        if operation == "narrow":
            operation = "widen" if after.measure > before.measure + EPS else "noop"
        return Refinement(operation, before, after, angle)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "window_size": self.window_size,
            "frozen": self.frozen,
            "history": list(self._history),
            "cones": self.cone_set.to_dict(),
        }


def make_learner(policy: str, window_size: int | None = None) -> NarrowingLearner:
    if policy == "window64":
        return NarrowingLearner("window64", window_size=window_size or 64)
    if policy.startswith("window"):
        size = window_size
        if size is None:
            size = int(policy.replace("window", ""))
        return NarrowingLearner(policy, window_size=size)
    if policy == "cumulative":
        return NarrowingLearner("cumulative", window_size=None)
    if policy == "frozen_a1":
        return NarrowingLearner("frozen_a1", window_size=None)
    if policy == "full_circle":
        return NarrowingLearner("full_circle", window_size=None)
    raise ValueError(f"unknown policy: {policy}")


def support_cone(supports: Mapping[str, Mapping[str, float]], key: str) -> ConeSet:
    return oracle_support([supports[key]])


def integer_support_angles(
    support: Mapping[str, float],
    resolution: int = 1,
) -> list[int]:
    center = float(support["center"])
    half_width = float(support["half_width"])
    low = int(round((center - half_width) / resolution))
    high = int(round((center + half_width) / resolution))
    return [int(normalize_angle(degree * resolution)) for degree in range(low, high + 1)]


def exact_support_coverage(
    cone_set: ConeSet,
    support: Mapping[str, float],
    resolution: int = 1,
) -> float:
    angles = integer_support_angles(support, resolution)
    if not angles:
        return 1.0
    return sum(1.0 for angle in angles if cone_set.contains(angle)) / len(angles)


__all__ = [
    "CIRCLE",
    "LEARNERS",
    "SITUATION_LABEL",
    "AngularCone",
    "Commitment",
    "ConeSet",
    "NarrowingLearner",
    "Refinement",
    "classify_operation",
    "exact_support_coverage",
    "excess_measure",
    "integer_support_angles",
    "make_learner",
    "normalize_angle",
    "rebuild_from_history",
    "support_cone",
]
