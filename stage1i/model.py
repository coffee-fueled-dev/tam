"""Stage 1I window multi-cone rebuild with mode pruning."""

from __future__ import annotations

from collections import deque
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
    minimal_covering_cone,
    normalize_angle,
    oracle_support,
)

LEARNERS = ("window_multi", "window_single", "cumulative_multi", "full_circle")
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
    before_count = len(before.cones)
    after_count = len(after.cones)
    if (
        abs(before.measure - after.measure) < EPS
        and before_count == after_count
        and cone_contains_cone(before, after)
        and cone_contains_cone(after, before)
    ):
        return "noop"
    if before_count == 0 and after_count > 0:
        return "add"
    if after_count < before_count and after.measure + EPS < before.measure:
        return "prune"
    if after_count > before_count:
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
        return "narrow" if after_count >= before_count else "prune"
    return "shift"


def cluster_angles(
    angles: Sequence[float],
    split_gap: float,
) -> list[list[float]]:
    points = sorted({normalize_angle(angle) for angle in angles})
    if not points:
        return []
    if len(points) == 1:
        return [points]

    n = len(points)
    gaps = [points[index + 1] - points[index] for index in range(n - 1)]
    gaps.append(points[0] + CIRCLE - points[-1])
    start = max(range(n), key=lambda index: gaps[index])
    order = [points[(start + 1 + offset) % n] for offset in range(n)]
    clusters: list[list[float]] = [[order[0]]]
    for offset in range(n - 1):
        gap = gaps[(start + 1 + offset) % n]
        if gap > split_gap + EPS:
            clusters.append([order[offset + 1]])
        else:
            clusters[-1].append(order[offset + 1])
    return clusters


def rebuild_multi_from_history(
    history: Sequence[float],
    split_gap: float,
) -> ConeSet:
    if not history:
        return ConeSet(())
    clusters = cluster_angles(history, split_gap)
    cones = [minimal_covering_cone(cluster) for cluster in clusters if cluster]
    return ConeSet(_canonical_cones(cones))


def rebuild_single_from_history(history: Sequence[float]) -> ConeSet:
    if not history:
        return ConeSet(())
    return ConeSet((minimal_covering_cone(history),))


class PruningLearner:
    """Geometric cone learner with window multi/single or cumulative multi."""

    def __init__(
        self,
        policy: str,
        window_size: int | None = 64,
        split_gap: float = 30.0,
    ) -> None:
        if policy not in LEARNERS and not (
            policy.startswith("window_multi") or policy.startswith("window_single")
        ):
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.window_size = window_size
        self.split_gap = float(split_gap)
        self.frozen = False
        self._history: Deque[float] | list[float]
        if policy.startswith("window") and window_size is not None:
            self._history = deque(maxlen=int(window_size))
        else:
            self._history = []
        self._cones: list[AngularCone] = []
        self._cumulative: GeometricLearner | None = None
        if policy == "cumulative_multi":
            self._cumulative = GeometricLearner("multi_cone", split_gap=split_gap)
        if policy == "full_circle":
            self._cones = [AngularCone(0.0, CIRCLE / 2)]

    @property
    def cone_set(self) -> ConeSet:
        if self.policy == "full_circle":
            return ConeSet((AngularCone(0.0, CIRCLE / 2),))
        if self._cumulative is not None:
            return self._cumulative.cone_set
        return ConeSet(tuple(self._cones))

    def predict(self, step: int) -> Commitment:
        return Commitment(self.cone_set, step)

    def freeze(self) -> None:
        self.frozen = True

    def observe(self, angle: float) -> Refinement:
        angle = normalize_angle(angle)
        before = self.cone_set

        if self.policy == "full_circle":
            return Refinement("noop", before, before, angle)

        if self.frozen:
            return Refinement("noop", before, before, angle)

        if self._cumulative is not None:
            refinement = self._cumulative.observe(angle)
            # Map stage1f ops; never prune.
            return refinement

        self._history.append(angle)
        history = list(self._history)
        if self.policy.startswith("window_multi"):
            after = rebuild_multi_from_history(history, self.split_gap)
        else:
            after = rebuild_single_from_history(history)
        self._cones = list(after.cones)
        operation = classify_operation(before, after)
        assert after.contains(angle)
        return Refinement(operation, before, after, angle)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "window_size": self.window_size,
            "split_gap": self.split_gap,
            "frozen": self.frozen,
            "history": list(self._history),
            "cones": self.cone_set.to_dict(),
        }


def make_learner(
    policy: str,
    window_size: int | None = None,
    split_gap: float = 30.0,
) -> PruningLearner:
    if policy == "window_multi":
        return PruningLearner(
            "window_multi",
            window_size=window_size or 64,
            split_gap=split_gap,
        )
    if policy == "window_single":
        return PruningLearner(
            "window_single",
            window_size=window_size or 64,
            split_gap=split_gap,
        )
    if policy.startswith("window_multi"):
        size = window_size or int(policy.replace("window_multi", "") or 64)
        return PruningLearner("window_multi", window_size=size, split_gap=split_gap)
    if policy.startswith("window_single"):
        size = window_size or int(policy.replace("window_single", "") or 64)
        return PruningLearner("window_single", window_size=size, split_gap=split_gap)
    if policy == "cumulative_multi":
        return PruningLearner("cumulative_multi", window_size=None, split_gap=split_gap)
    if policy == "full_circle":
        return PruningLearner("full_circle", window_size=None, split_gap=split_gap)
    raise ValueError(f"unknown policy: {policy}")


def integer_support_angles(
    modes: Sequence[Mapping[str, float]],
    resolution: int = 1,
) -> list[int]:
    angles: set[int] = set()
    for mode in modes:
        center = float(mode["center"])
        half_width = float(mode["half_width"])
        low = int(round((center - half_width) / resolution))
        high = int(round((center + half_width) / resolution))
        for degree in range(low, high + 1):
            angles.add(int(normalize_angle(degree * resolution)))
    return sorted(angles)


def exact_support_coverage(
    cone_set: ConeSet,
    modes: Sequence[Mapping[str, float]],
    resolution: int = 1,
) -> float:
    angles = integer_support_angles(modes, resolution)
    if not angles:
        return 1.0
    return sum(1.0 for angle in angles if cone_set.contains(angle)) / len(angles)


def support_cone(modes: Sequence[Mapping[str, float]]) -> ConeSet:
    return oracle_support(list(modes))


def regime_modes(
    protocol: Mapping[str, object],
    regime: str,
) -> list[dict[str, float]]:
    modes = protocol["modes"]
    assert isinstance(modes, Mapping)
    regimes = protocol["regimes"]
    assert isinstance(regimes, Mapping)
    names = regimes[regime]
    assert isinstance(names, Sequence)
    return [dict(modes[name]) for name in names]  # type: ignore[index]


__all__ = [
    "CIRCLE",
    "LEARNERS",
    "SITUATION_LABEL",
    "AngularCone",
    "Commitment",
    "ConeSet",
    "PruningLearner",
    "Refinement",
    "circular_distance",
    "classify_operation",
    "cluster_angles",
    "exact_support_coverage",
    "excess_measure",
    "make_learner",
    "normalize_angle",
    "rebuild_multi_from_history",
    "rebuild_single_from_history",
    "regime_modes",
    "support_cone",
]
