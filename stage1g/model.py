"""Stage 1G situation-conditioned geometric cone learners."""

from __future__ import annotations

from typing import Mapping, Sequence

from stage1f.model import (
    CIRCLE,
    AngularCone,
    Commitment,
    ConeSet,
    GeometricLearner,
    Refinement,
    excess_measure,
    normalize_angle,
    oracle_support,
)

SITUATIONS = ("tight", "wide", "shifted")
LEARNERS = (
    "unconditional_multi",
    "situation_multi",
    "unconditional_single",
    "full_circle",
)


class SituationalLearner:
    """Route commitments/updates through global or per-situation cone stores."""

    def __init__(
        self,
        policy: str,
        situations: Sequence[str] = SITUATIONS,
        split_gap: float = 30.0,
    ) -> None:
        if policy not in LEARNERS:
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.situations = tuple(situations)
        self.split_gap = float(split_gap)
        self.frozen = False

        if policy == "situation_multi":
            self._stores = {
                situation: GeometricLearner("multi_cone", split_gap=split_gap)
                for situation in self.situations
            }
        elif policy == "unconditional_multi":
            store = GeometricLearner("multi_cone", split_gap=split_gap)
            self._stores = {situation: store for situation in self.situations}
            self._global = store
        elif policy == "unconditional_single":
            store = GeometricLearner("single_widen", split_gap=split_gap)
            self._stores = {situation: store for situation in self.situations}
            self._global = store
        else:
            store = GeometricLearner("full_circle", split_gap=split_gap)
            self._stores = {situation: store for situation in self.situations}
            self._global = store

    def _store(self, situation: str) -> GeometricLearner:
        if situation not in self._stores:
            raise ValueError(f"unknown situation: {situation}")
        return self._stores[situation]

    def predict(self, situation: str, step: int) -> Commitment:
        return self._store(situation).predict(step)

    def observe(self, situation: str, angle: float) -> Refinement:
        if self.frozen:
            before = self._store(situation).cone_set
            return Refinement("noop", before, before, normalize_angle(angle))
        return self._store(situation).observe(angle)

    def freeze(self) -> None:
        self.frozen = True

    def cone_set(self, situation: str) -> ConeSet:
        return self._store(situation).cone_set

    def to_dict(self) -> dict[str, object]:
        if self.policy == "situation_multi":
            stores = {
                situation: self._stores[situation].to_dict()
                for situation in self.situations
            }
        else:
            stores = {"global": self._global.to_dict()}
        return {
            "policy": self.policy,
            "frozen": self.frozen,
            "split_gap": self.split_gap,
            "stores": stores,
        }


def support_for(
    supports: Mapping[str, Mapping[str, float]],
    situation: str,
) -> ConeSet:
    mode = supports[situation]
    return oracle_support([mode])


def situation_excess(
    learned: ConeSet,
    supports: Mapping[str, Mapping[str, float]],
    situation: str,
) -> float:
    return excess_measure(learned, support_for(supports, situation))


def dominant_center(cone_set: ConeSet) -> float | None:
    if not cone_set.cones:
        return None
    cone = max(cone_set.cones, key=lambda item: item.measure)
    return cone.center


__all__ = [
    "CIRCLE",
    "LEARNERS",
    "SITUATIONS",
    "AngularCone",
    "Commitment",
    "ConeSet",
    "Refinement",
    "SituationalLearner",
    "dominant_center",
    "normalize_angle",
    "situation_excess",
    "support_for",
]
