"""Stage 1F literal angular cones over directional trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

CIRCLE = 360.0
EPS = 1e-9


def normalize_angle(angle: float) -> float:
    return float(angle) % CIRCLE


def circular_distance(a: float, b: float) -> float:
    delta = abs(normalize_angle(a) - normalize_angle(b))
    return min(delta, CIRCLE - delta)


@dataclass(frozen=True)
class AngularCone:
    center: float
    half_width: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", normalize_angle(self.center))
        object.__setattr__(self, "half_width", float(self.half_width))
        if self.half_width < -EPS:
            raise ValueError("half_width must be non-negative")
        if self.half_width > CIRCLE / 2 + EPS:
            object.__setattr__(self, "half_width", CIRCLE / 2)

    @property
    def measure(self) -> float:
        return min(CIRCLE, 2.0 * self.half_width)

    def contains(self, angle: float) -> bool:
        if self.half_width >= CIRCLE / 2 - EPS:
            return True
        return circular_distance(angle, self.center) <= self.half_width + EPS

    def exterior_distance(self, angle: float) -> float:
        if self.contains(angle):
            return 0.0
        return circular_distance(angle, self.center) - self.half_width

    def to_dict(self) -> dict[str, float]:
        return {
            "center": self.center,
            "half_width": self.half_width,
            "measure": self.measure,
        }


def minimal_covering_cone(angles: Iterable[float]) -> AngularCone:
    points = sorted({normalize_angle(angle) for angle in angles})
    if not points:
        raise ValueError("cannot cover an empty angle set")
    if len(points) == 1:
        return AngularCone(points[0], 0.0)

    gaps: list[tuple[float, int]] = []
    for index in range(len(points) - 1):
        gaps.append((points[index + 1] - points[index], index))
    gaps.append((points[0] + CIRCLE - points[-1], len(points) - 1))
    max_gap, gap_index = max(gaps)
    start = points[(gap_index + 1) % len(points)]
    measure = CIRCLE - max_gap
    if measure >= CIRCLE - EPS:
        return AngularCone(0.0, CIRCLE / 2)
    center = normalize_angle(start + measure / 2.0)
    return AngularCone(center, measure / 2.0)


def merge_cones(left: AngularCone, right: AngularCone) -> AngularCone:
    if left.half_width >= CIRCLE / 2 - EPS:
        return AngularCone(left.center, CIRCLE / 2)
    if right.half_width >= CIRCLE / 2 - EPS:
        return AngularCone(right.center, CIRCLE / 2)
    samples = [
        normalize_angle(left.center - left.half_width),
        normalize_angle(left.center + left.half_width),
        normalize_angle(right.center - right.half_width),
        normalize_angle(right.center + right.half_width),
    ]
    # Dense enough endpoints: also include centers for point cones.
    samples.extend([left.center, right.center])
    return minimal_covering_cone(samples)


def cones_overlap_or_touch(left: AngularCone, right: AngularCone) -> bool:
    return circular_distance(left.center, right.center) <= (
        left.half_width + right.half_width + EPS
    )


@dataclass(frozen=True)
class ConeSet:
    cones: tuple[AngularCone, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cones", tuple(self.cones))

    def contains(self, angle: float) -> bool:
        return any(cone.contains(angle) for cone in self.cones)

    @property
    def measure(self) -> float:
        if not self.cones:
            return 0.0
        # Sweep-line union on the circle via unrolled intervals.
        intervals: list[tuple[float, float]] = []
        for cone in self.cones:
            if cone.half_width >= CIRCLE / 2 - EPS:
                return CIRCLE
            start = normalize_angle(cone.center - cone.half_width)
            width = cone.measure
            end = start + width
            if end <= CIRCLE + EPS:
                intervals.append((start, min(end, CIRCLE)))
            else:
                intervals.append((start, CIRCLE))
                intervals.append((0.0, end - CIRCLE))
        if not intervals:
            return 0.0
        intervals.sort()
        total = 0.0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end + EPS:
                cur_end = max(cur_end, end)
            else:
                total += cur_end - cur_start
                cur_start, cur_end = start, end
        total += cur_end - cur_start
        return min(CIRCLE, total)

    def to_dict(self) -> dict[str, object]:
        return {
            "cones": [cone.to_dict() for cone in self.cones],
            "measure": self.measure,
            "count": len(self.cones),
        }


FULL_CIRCLE = ConeSet((AngularCone(0.0, CIRCLE / 2),))


@dataclass(frozen=True)
class Commitment:
    cones: ConeSet
    step: int

    def contains(self, angle: float) -> bool:
        return self.cones.contains(angle)

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "cones": self.cones.to_dict(),
        }


@dataclass(frozen=True)
class Refinement:
    operation: str
    before: ConeSet
    after: ConeSet
    angle: float

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "angle": self.angle,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


def _canonical_cones(cones: Sequence[AngularCone]) -> tuple[AngularCone, ...]:
    cleaned = [
        cone if cone.half_width < CIRCLE / 2 - EPS else AngularCone(0.0, CIRCLE / 2)
        for cone in cones
    ]
    if any(cone.half_width >= CIRCLE / 2 - EPS for cone in cleaned):
        return (AngularCone(0.0, CIRCLE / 2),)

    remaining = list(cleaned)
    changed = True
    while changed and len(remaining) > 1:
        changed = False
        merged: list[AngularCone] = []
        used = [False] * len(remaining)
        for i, left in enumerate(remaining):
            if used[i]:
                continue
            current = left
            for j in range(i + 1, len(remaining)):
                if used[j]:
                    continue
                right = remaining[j]
                if cones_overlap_or_touch(current, right):
                    current = merge_cones(current, right)
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append(current)
        remaining = merged

    remaining.sort(key=lambda cone: (cone.center, cone.half_width))
    return tuple(remaining)


class GeometricLearner:
    """Cumulative geometric cone learner with widen / add refinement."""

    def __init__(self, policy: str, split_gap: float = 30.0) -> None:
        if policy not in ("single_widen", "multi_cone", "full_circle"):
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.split_gap = float(split_gap)
        self._cones: list[AngularCone] = []
        if policy == "full_circle":
            self._cones = [AngularCone(0.0, CIRCLE / 2)]

    @property
    def cone_set(self) -> ConeSet:
        return ConeSet(tuple(self._cones))

    def predict(self, step: int) -> Commitment:
        return Commitment(self.cone_set, step)

    def observe(self, angle: float) -> Refinement:
        angle = normalize_angle(angle)
        before = self.cone_set

        if self.policy == "full_circle":
            after = FULL_CIRCLE
            self._cones = list(after.cones)
            return Refinement("noop", before, after, angle)

        if before.contains(angle):
            return Refinement("noop", before, before, angle)

        if not self._cones:
            self._cones = [AngularCone(angle, 0.0)]
            after = self.cone_set
            return Refinement("add", before, after, angle)

        if self.policy == "single_widen":
            # Minimal connected sector covering prior cone endpoints and the miss.
            endpoints = [
                normalize_angle(self._cones[0].center - self._cones[0].half_width),
                normalize_angle(self._cones[0].center + self._cones[0].half_width),
                self._cones[0].center,
                angle,
            ]
            self._cones = [minimal_covering_cone(endpoints)]
            after = self.cone_set
            assert after.contains(angle)
            return Refinement("widen", before, after, angle)

        # multi_cone
        nearest_index = min(
            range(len(self._cones)),
            key=lambda index: (
                self._cones[index].exterior_distance(angle),
                self._cones[index].center,
                index,
            ),
        )
        nearest = self._cones[nearest_index]
        if nearest.exterior_distance(angle) <= self.split_gap + EPS:
            endpoints = [
                normalize_angle(nearest.center - nearest.half_width),
                normalize_angle(nearest.center + nearest.half_width),
                nearest.center,
                angle,
            ]
            self._cones[nearest_index] = minimal_covering_cone(endpoints)
            operation = "widen"
        else:
            self._cones.append(AngularCone(angle, 0.0))
            operation = "add"

        self._cones = list(_canonical_cones(self._cones))
        after = self.cone_set
        assert after.contains(angle)
        return Refinement(operation, before, after, angle)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "split_gap": self.split_gap,
            "cones": self.cone_set.to_dict(),
        }


def oracle_support(modes: Sequence[Mapping[str, float]]) -> ConeSet:
    cones = [
        AngularCone(float(mode["center"]), float(mode["half_width"]))
        for mode in modes
    ]
    return ConeSet(_canonical_cones(cones))


def excess_measure(learned: ConeSet, support: ConeSet) -> float:
    """Angular measure in the learned set but outside oracle support."""
    return max(0.0, learned.measure - support.measure)


def mode_represented(cone_set: ConeSet, mode: Mapping[str, float]) -> bool:
    center = float(mode["center"])
    half_width = float(mode["half_width"])
    # Represented if the mode center and both endpoints lie in the cone set.
    samples = (
        center,
        normalize_angle(center - half_width),
        normalize_angle(center + half_width),
    )
    return all(cone_set.contains(sample) for sample in samples)
