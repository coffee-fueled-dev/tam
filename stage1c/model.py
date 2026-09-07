"""Stage 1C context-pooled predictor and commitment/selection ablations."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

PORTS = ("n", "e", "s", "w", "stay")
PORT_DELTAS = {
    "n": (0, 1),
    "e": (1, 0),
    "s": (0, -1),
    "w": (-1, 0),
    "stay": (0, 0),
}
ROTATED_DELTAS = {
    "n": (1, 0),
    "e": (0, -1),
    "s": (-1, 0),
    "w": (0, 1),
    "stay": (0, 0),
}
OUTCOMES = ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0))
# MAP and ranking ties prefer this displacement order.
MAP_OUTCOME_ORDER = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
TERRAINS = ("plain", "rotated")
CONTROLLERS = ("tam_cone", "expected_value", "point_map", "full_domain")


@dataclass(frozen=True)
class Observation:
    position: tuple[int, int]
    target: tuple[int, int]
    terrain: str


@dataclass(frozen=True)
class Commitment:
    probabilities: Mapping[tuple[int, int], float]
    cone: tuple[tuple[int, int], ...]


def _outcome_key(outcome: tuple[int, int]) -> str:
    return f"{outcome[0]},{outcome[1]}"


def _parse_outcome(key: str) -> tuple[int, int]:
    dx, dy = key.split(",")
    return int(dx), int(dy)


def probabilities_from_history(history: deque) -> dict[tuple[int, int], float]:
    counts = Counter(history)
    denominator = len(history) + 0.5 * len(OUTCOMES)
    return {
        outcome: (counts[outcome] + 0.5) / denominator
        for outcome in OUTCOMES
    }


def cone_90(probabilities: Mapping[tuple[int, int], float]) -> tuple[tuple[int, int], ...]:
    ranked = sorted(
        OUTCOMES,
        key=lambda outcome: (-probabilities[outcome], MAP_OUTCOME_ORDER.index(outcome)),
    )
    cone: list[tuple[int, int]] = []
    mass = 0.0
    for outcome in ranked:
        cone.append(outcome)
        mass += probabilities[outcome]
        if mass >= 0.9:
            break
    return tuple(cone)


def map_outcome(probabilities: Mapping[tuple[int, int], float]) -> tuple[int, int]:
    return min(
        OUTCOMES,
        key=lambda outcome: (-probabilities[outcome], MAP_OUTCOME_ORDER.index(outcome)),
    )


def full_domain_cone() -> tuple[tuple[int, int], ...]:
    return OUTCOMES


def build_commitment(
    probabilities: Mapping[tuple[int, int], float],
    controller: str,
) -> Commitment:
    if controller == "tam_cone":
        cone = cone_90(probabilities)
    elif controller == "expected_value":
        # Post-hoc 90% set for measurement only; selection ignores cone width.
        cone = cone_90(probabilities)
    elif controller == "point_map":
        cone = (map_outcome(probabilities),)
    elif controller == "full_domain":
        cone = full_domain_cone()
    else:
        raise ValueError(f"unknown controller: {controller}")
    return Commitment(MappingProxyType(dict(probabilities)), cone)


def expected_squared_distance(
    observation: Observation,
    probabilities: Mapping[tuple[int, int], float],
) -> float:
    return sum(
        probability
        * (
            (observation.position[0] + dx - observation.target[0]) ** 2
            + (observation.position[1] + dy - observation.target[1]) ** 2
        )
        for (dx, dy), probability in probabilities.items()
    )


def map_squared_distance(
    observation: Observation,
    displacement: tuple[int, int],
) -> float:
    dx, dy = displacement
    return (
        (observation.position[0] + dx - observation.target[0]) ** 2
        + (observation.position[1] + dy - observation.target[1]) ** 2
    )


def select_port(
    observation: Observation,
    port_probabilities: Mapping[str, Mapping[tuple[int, int], float]],
    controller: str,
    rng,
) -> tuple[str, Commitment]:
    commitments = {
        port: build_commitment(probabilities, controller)
        for port, probabilities in port_probabilities.items()
    }
    if controller == "point_map":
        losses = {
            port: map_squared_distance(observation, commitments[port].cone[0])
            for port in PORTS
        }
    else:
        losses = {
            port: expected_squared_distance(
                observation, commitments[port].probabilities
            )
            for port in PORTS
        }
    best_loss = min(losses.values())
    candidates = [
        port for port in PORTS if abs(losses[port] - best_loss) < 1e-12
    ]
    if controller == "tam_cone":
        smallest = min(len(commitments[port].cone) for port in candidates)
        candidates = [
            port for port in candidates if len(commitments[port].cone) == smallest
        ]
        port = rng.choice(candidates)
    elif controller == "point_map":
        # Remaining ties: higher MAP probability, then stable port order.
        best_map = max(
            commitments[port].probabilities[commitments[port].cone[0]]
            for port in candidates
        )
        candidates = [
            port for port in candidates
            if abs(
                commitments[port].probabilities[commitments[port].cone[0]]
                - best_map
            ) < 1e-12
        ]
        port = sorted(candidates, key=PORTS.index)[0]
    else:
        port = rng.choice(candidates)
    return port, commitments[port]


class ContextPooledPredictor:
    """One outcome distribution per (terrain, port)."""

    def __init__(self, history_size: int = 64) -> None:
        self.history_size = history_size
        self._histories = {
            (terrain, port): deque(maxlen=history_size)
            for terrain in TERRAINS
            for port in PORTS
        }

    def probabilities(self, observation: Observation, port: str) -> dict[tuple[int, int], float]:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        if observation.terrain not in TERRAINS:
            raise ValueError(f"unknown terrain: {observation.terrain}")
        return probabilities_from_history(
            self._histories[(observation.terrain, port)]
        )

    def predict(self, observation: Observation, port: str) -> Commitment:
        return build_commitment(self.probabilities(observation, port), "tam_cone")

    def observe(
        self,
        observation: Observation,
        port: str,
        displacement: tuple[int, int],
    ) -> None:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        if observation.terrain not in TERRAINS:
            raise ValueError(f"unknown terrain: {observation.terrain}")
        if displacement not in OUTCOMES:
            raise ValueError(f"unsupported displacement: {displacement}")
        self._histories[(observation.terrain, port)].append(displacement)

    def clear(self) -> None:
        for history in self._histories.values():
            history.clear()

    def sample_count(self) -> int:
        return sum(len(history) for history in self._histories.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "context_pooled",
            "history_size": self.history_size,
            "histories": {
                f"{terrain}:{port}": [
                    _outcome_key(value) for value in self._histories[(terrain, port)]
                ]
                for terrain in TERRAINS
                for port in PORTS
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ContextPooledPredictor":
        model = cls(int(data["history_size"]))
        histories = data["histories"]
        if not isinstance(histories, Mapping):
            raise ValueError("histories must be a mapping")
        for key, values in histories.items():
            terrain, port = str(key).split(":", 1)
            for value in values:
                model._histories[(terrain, port)].append(_parse_outcome(str(value)))
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ContextPooledPredictor":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def brier_score(
    probabilities: Mapping[tuple[int, int], float],
    outcome: tuple[int, int],
) -> float:
    return sum(
        (probabilities[value] - (1.0 if value == outcome else 0.0)) ** 2
        for value in OUTCOMES
    )


def commitment_brier(
    commitment: Commitment,
    outcome: tuple[int, int],
) -> float:
    """Brier score after renormalizing predictive mass onto the committed cone."""
    mass = sum(commitment.probabilities[outcome_] for outcome_ in commitment.cone)
    if mass <= 0:
        renormalized = {value: 1 / len(OUTCOMES) for value in OUTCOMES}
    else:
        renormalized = {
            value: (
                commitment.probabilities[value] / mass
                if value in commitment.cone
                else 0.0
            )
            for value in OUTCOMES
        }
    return brier_score(renormalized, outcome)
