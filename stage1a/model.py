"""Stage 1A 2D trajectory-affordance learner (context-independent)."""

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
OUTCOMES = ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0))
MOVING_PORTS = ("n", "e", "s", "w")


@dataclass(frozen=True)
class Observation:
    position: tuple[int, int]
    target: tuple[int, int]


@dataclass(frozen=True)
class Commitment:
    probabilities: Mapping[tuple[int, int], float]
    cone: tuple[tuple[int, int], ...]


def _outcome_key(outcome: tuple[int, int]) -> str:
    return f"{outcome[0]},{outcome[1]}"


def _parse_outcome(key: str) -> tuple[int, int]:
    dx, dy = key.split(",")
    return int(dx), int(dy)


class OutcomeModel:
    """Bounded empirical joint-displacement model with Dirichlet smoothing."""

    def __init__(self, history_size: int = 32) -> None:
        self.history_size = history_size
        self._histories = {
            port: deque(maxlen=history_size) for port in PORTS
        }

    def predict(self, port: str) -> Commitment:
        self._check_port(port)
        history = self._histories[port]
        counts = Counter(history)
        denominator = len(history) + 0.5 * len(OUTCOMES)
        probabilities = {
            outcome: (counts[outcome] + 0.5) / denominator
            for outcome in OUTCOMES
        }
        ranked = sorted(
            OUTCOMES,
            key=lambda outcome: (-probabilities[outcome], outcome),
        )
        cone: list[tuple[int, int]] = []
        mass = 0.0
        for outcome in ranked:
            cone.append(outcome)
            mass += probabilities[outcome]
            if mass >= 0.9:
                break
        return Commitment(MappingProxyType(probabilities), tuple(cone))

    def observe(self, port: str, displacement: tuple[int, int]) -> None:
        self._check_port(port)
        if displacement not in OUTCOMES:
            raise ValueError(f"unsupported displacement: {displacement}")
        self._histories[port].append(displacement)

    def clear(self) -> None:
        for history in self._histories.values():
            history.clear()

    def to_dict(self) -> dict[str, object]:
        return {
            "history_size": self.history_size,
            "histories": {
                port: [_outcome_key(value) for value in self._histories[port]]
                for port in PORTS
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OutcomeModel":
        model = cls(int(data["history_size"]))
        histories = data["histories"]
        if not isinstance(histories, Mapping):
            raise ValueError("histories must be a mapping")
        for port in PORTS:
            values = histories.get(port, [])
            if not isinstance(values, list):
                raise ValueError(f"history for {port} must be a list")
            for value in values:
                model.observe(port, _parse_outcome(str(value)))
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "OutcomeModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def _check_port(port: str) -> None:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")


class Agent:
    def __init__(
        self,
        model: OutcomeModel | None = None,
        exploration: float = 0.1,
    ) -> None:
        self.model = model or OutcomeModel()
        self.exploration = exploration

    def select(self, observation: Observation, rng) -> tuple[str, Commitment]:
        predictions = {port: self.model.predict(port) for port in PORTS}
        if rng.random() < self.exploration:
            port = rng.choice(PORTS)
            return port, predictions[port]

        losses = {
            port: sum(
                probability
                * (
                    (observation.position[0] + dx - observation.target[0]) ** 2
                    + (observation.position[1] + dy - observation.target[1]) ** 2
                )
                for (dx, dy), probability in prediction.probabilities.items()
            )
            for port, prediction in predictions.items()
        }
        best_loss = min(losses.values())
        candidates = [
            port for port in PORTS if abs(losses[port] - best_loss) < 1e-12
        ]
        smallest_cone = min(len(predictions[port].cone) for port in candidates)
        candidates = [
            port for port in candidates
            if len(predictions[port].cone) == smallest_cone
        ]
        port = rng.choice(candidates)
        return port, predictions[port]

    def observe(self, port: str, displacement: tuple[int, int]) -> None:
        self.model.observe(port, displacement)


def brier_score(
    probabilities: Mapping[tuple[int, int], float],
    outcome: tuple[int, int],
) -> float:
    return sum(
        (probabilities[value] - (1.0 if value == outcome else 0.0)) ** 2
        for value in OUTCOMES
    )


def squared_distance(
    position: tuple[int, int],
    target: tuple[int, int],
) -> int:
    return (position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2
