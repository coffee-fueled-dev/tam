"""Stage 1B predictors: unconditional, exact-state, and context-pooled."""

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
# 90-degree clockwise rotation of the plain displacement.
ROTATED_DELTAS = {
    "n": (1, 0),
    "e": (0, -1),
    "s": (-1, 0),
    "w": (0, 1),
    "stay": (0, 0),
}
OUTCOMES = ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0))
TERRAINS = ("plain", "rotated")
PREDICTORS = ("unconditional", "exact_state", "context_pooled")


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


def _commitment_from_history(history: deque) -> Commitment:
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


class BasePredictor:
    history_size: int

    def predict(self, observation: Observation, port: str) -> Commitment:
        raise NotImplementedError

    def observe(
        self,
        observation: Observation,
        port: str,
        displacement: tuple[int, int],
    ) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError

    def sample_count(self) -> int:
        raise NotImplementedError


class UnconditionalPredictor(BasePredictor):
    """One outcome distribution per port."""

    def __init__(self, history_size: int = 32) -> None:
        self.history_size = history_size
        self._histories = {
            port: deque(maxlen=history_size) for port in PORTS
        }

    def predict(self, observation: Observation, port: str) -> Commitment:
        self._check_port(port)
        return _commitment_from_history(self._histories[port])

    def observe(
        self,
        observation: Observation,
        port: str,
        displacement: tuple[int, int],
    ) -> None:
        self._check_port(port)
        if displacement not in OUTCOMES:
            raise ValueError(f"unsupported displacement: {displacement}")
        self._histories[port].append(displacement)

    def clear(self) -> None:
        for history in self._histories.values():
            history.clear()

    def sample_count(self) -> int:
        return sum(len(history) for history in self._histories.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "unconditional",
            "history_size": self.history_size,
            "histories": {
                port: [_outcome_key(value) for value in self._histories[port]]
                for port in PORTS
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "UnconditionalPredictor":
        model = cls(int(data["history_size"]))
        histories = data["histories"]
        assert isinstance(histories, Mapping)
        for port in PORTS:
            for value in histories.get(port, []):
                model._histories[port].append(_parse_outcome(str(value)))
        return model

    @staticmethod
    def _check_port(port: str) -> None:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")


class ExactStatePredictor(BasePredictor):
    """One outcome distribution per (x, y, port); no transfer."""

    def __init__(self, history_size: int = 32) -> None:
        self.history_size = history_size
        self._histories: dict[tuple[int, int, str], deque] = {}

    def _key(self, observation: Observation, port: str) -> tuple[int, int, str]:
        return (observation.position[0], observation.position[1], port)

    def _history(self, key: tuple[int, int, str]) -> deque:
        if key not in self._histories:
            self._histories[key] = deque(maxlen=self.history_size)
        return self._histories[key]

    def predict(self, observation: Observation, port: str) -> Commitment:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        return _commitment_from_history(self._history(self._key(observation, port)))

    def observe(
        self,
        observation: Observation,
        port: str,
        displacement: tuple[int, int],
    ) -> None:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        if displacement not in OUTCOMES:
            raise ValueError(f"unsupported displacement: {displacement}")
        self._history(self._key(observation, port)).append(displacement)

    def clear(self) -> None:
        self._histories.clear()

    def sample_count(self) -> int:
        return sum(len(history) for history in self._histories.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "exact_state",
            "history_size": self.history_size,
            "histories": {
                f"{x},{y},{port}": [_outcome_key(value) for value in history]
                for (x, y, port), history in sorted(self._histories.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExactStatePredictor":
        model = cls(int(data["history_size"]))
        histories = data["histories"]
        assert isinstance(histories, Mapping)
        for key, values in histories.items():
            x_text, y_text, port = str(key).split(",", 2)
            history_key = (int(x_text), int(y_text), port)
            history = model._history(history_key)
            for value in values:
                history.append(_parse_outcome(str(value)))
        return model


class ContextPooledPredictor(BasePredictor):
    """One outcome distribution per (terrain, port)."""

    def __init__(self, history_size: int = 32) -> None:
        self.history_size = history_size
        self._histories = {
            (terrain, port): deque(maxlen=history_size)
            for terrain in TERRAINS
            for port in PORTS
        }

    def predict(self, observation: Observation, port: str) -> Commitment:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        if observation.terrain not in TERRAINS:
            raise ValueError(f"unknown terrain: {observation.terrain}")
        return _commitment_from_history(
            self._histories[(observation.terrain, port)]
        )

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
        assert isinstance(histories, Mapping)
        for key, values in histories.items():
            terrain, port = str(key).split(":", 1)
            for value in values:
                model._histories[(terrain, port)].append(_parse_outcome(str(value)))
        return model


def make_predictor(kind: str, history_size: int = 32) -> BasePredictor:
    if kind == "unconditional":
        return UnconditionalPredictor(history_size)
    if kind == "exact_state":
        return ExactStatePredictor(history_size)
    if kind == "context_pooled":
        return ContextPooledPredictor(history_size)
    raise ValueError(f"unknown predictor: {kind}")


def predictor_from_dict(data: Mapping[str, object]) -> BasePredictor:
    kind = str(data["kind"])
    if kind == "unconditional":
        return UnconditionalPredictor.from_dict(data)
    if kind == "exact_state":
        return ExactStatePredictor.from_dict(data)
    if kind == "context_pooled":
        return ContextPooledPredictor.from_dict(data)
    raise ValueError(f"unknown predictor kind: {kind}")


class Agent:
    def __init__(
        self,
        predictor: BasePredictor,
        exploration: float = 0.1,
    ) -> None:
        self.predictor = predictor
        self.exploration = exploration

    def select(self, observation: Observation, rng) -> tuple[str, Commitment]:
        predictions = {
            port: self.predictor.predict(observation, port) for port in PORTS
        }
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

    def observe(
        self,
        observation: Observation,
        port: str,
        displacement: tuple[int, int],
    ) -> None:
        self.predictor.observe(observation, port, displacement)


def brier_score(
    probabilities: Mapping[tuple[int, int], float],
    outcome: tuple[int, int],
) -> float:
    return sum(
        (probabilities[value] - (1.0 if value == outcome else 0.0)) ** 2
        for value in OUTCOMES
    )


def save_predictor(predictor: BasePredictor, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(predictor.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_predictor(path: str | Path) -> BasePredictor:
    return predictor_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
