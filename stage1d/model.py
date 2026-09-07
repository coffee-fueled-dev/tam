"""Stage 1D categorical predictor and asymmetric-risk selectors."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

PORTS = ("safe", "rare_tail")
LOSS_DOMAIN = (0.0, 2.0, 12.0, 14.4, 18.0)
CONTROLLERS = ("expected_value", "tam_90", "cvar_90", "worst_case")
CATASTROPHE_THRESHOLD = 4.0
COMMITMENT_MASS = 0.9
CVAR_TAIL = 0.1


@dataclass(frozen=True)
class Commitment:
    probabilities: Mapping[float, float]
    cone: tuple[float, ...]


def exact_probabilities(port: str, catastrophe_probability: float, catastrophe_loss: float) -> dict[float, float]:
    probabilities = {loss: 0.0 for loss in LOSS_DOMAIN}
    if port == "safe":
        probabilities[2.0] = 1.0
        return probabilities
    if port != "rare_tail":
        raise ValueError(f"unknown port: {port}")
    if catastrophe_loss not in LOSS_DOMAIN:
        raise ValueError(f"unsupported catastrophe loss: {catastrophe_loss}")
    probabilities[0.0] = 1.0 - catastrophe_probability
    probabilities[catastrophe_loss] = catastrophe_probability
    return probabilities


def cone_90(probabilities: Mapping[float, float]) -> tuple[float, ...]:
    ranked = sorted(
        LOSS_DOMAIN,
        key=lambda loss: (-probabilities.get(loss, 0.0), loss),
    )
    cone: list[float] = []
    mass = 0.0
    for loss in ranked:
        probability = probabilities.get(loss, 0.0)
        if probability <= 0.0:
            continue
        cone.append(loss)
        mass += probability
        if mass >= COMMITMENT_MASS:
            break
    if not cone:
        cone = [min(LOSS_DOMAIN)]
    return tuple(cone)


def build_commitment(probabilities: Mapping[float, float]) -> Commitment:
    return Commitment(MappingProxyType(dict(probabilities)), cone_90(probabilities))


def expected_loss(probabilities: Mapping[float, float]) -> float:
    return sum(loss * probabilities.get(loss, 0.0) for loss in LOSS_DOMAIN)


def maximum_supported_loss(
    probabilities: Mapping[float, float],
    support: tuple[float, ...] | None = None,
) -> float:
    if support is None:
        supported = [loss for loss in LOSS_DOMAIN if probabilities.get(loss, 0.0) > 0.0]
    else:
        supported = list(support)
    if not supported:
        return max(LOSS_DOMAIN)
    return max(supported)


def cvar_90(probabilities: Mapping[float, float]) -> float:
    """Mean loss in the worst 10% probability mass (upper CVaR)."""
    remaining = CVAR_TAIL
    total = 0.0
    for loss in sorted(LOSS_DOMAIN, reverse=True):
        probability = probabilities.get(loss, 0.0)
        if probability <= 0.0:
            continue
        take = min(probability, remaining)
        total += loss * take
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / CVAR_TAIL


def cone_is_admissible(cone: tuple[float, ...]) -> bool:
    return all(loss <= CATASTROPHE_THRESHOLD for loss in cone)


def select_port(
    port_probabilities: Mapping[str, Mapping[float, float]],
    controller: str,
    port_support: Mapping[str, tuple[float, ...]] | None = None,
) -> tuple[str, Commitment]:
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    commitments = {
        port: build_commitment(probabilities)
        for port, probabilities in port_probabilities.items()
    }
    if controller == "expected_value":
        scores = {port: expected_loss(commitments[port].probabilities) for port in PORTS}
        chosen = min(PORTS, key=lambda port: (scores[port], PORTS.index(port)))
    elif controller == "cvar_90":
        scores = {port: cvar_90(commitments[port].probabilities) for port in PORTS}
        chosen = min(PORTS, key=lambda port: (scores[port], PORTS.index(port)))
    elif controller == "worst_case":
        scores = {}
        for port in PORTS:
            support = None if port_support is None else port_support.get(port)
            scores[port] = maximum_supported_loss(
                commitments[port].probabilities,
                support,
            )
        chosen = min(PORTS, key=lambda port: (scores[port], PORTS.index(port)))
    else:
        admissible = [
            port for port in PORTS if cone_is_admissible(commitments[port].cone)
        ]
        candidates = admissible if admissible else list(PORTS)
        scores = {
            port: expected_loss(commitments[port].probabilities)
            for port in candidates
        }
        chosen = min(candidates, key=lambda port: (scores[port], PORTS.index(port)))
    return chosen, commitments[chosen]


class LossModel:
    """Bounded categorical loss model with symmetric Dirichlet smoothing."""

    def __init__(self, history_size: int = 2000) -> None:
        self.history_size = history_size
        self._histories = {
            port: deque(maxlen=history_size) for port in PORTS
        }

    def probabilities(self, port: str) -> dict[float, float]:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        history = self._histories[port]
        counts = Counter(history)
        denominator = len(history) + 0.5 * len(LOSS_DOMAIN)
        return {
            loss: (counts[loss] + 0.5) / denominator
            for loss in LOSS_DOMAIN
        }

    def observe(self, port: str, loss: float) -> None:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        if loss not in LOSS_DOMAIN:
            raise ValueError(f"unsupported loss: {loss}")
        self._histories[port].append(loss)

    def clear(self) -> None:
        for history in self._histories.values():
            history.clear()

    def supported(self, port: str) -> tuple[float, ...]:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        return tuple(
            loss for loss in LOSS_DOMAIN if self._histories[port].count(loss) > 0
        )

    def sample_count(self) -> int:
        return sum(len(history) for history in self._histories.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "history_size": self.history_size,
            "histories": {
                port: list(self._histories[port]) for port in PORTS
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LossModel":
        model = cls(int(data["history_size"]))
        histories = data["histories"]
        if not isinstance(histories, Mapping):
            raise ValueError("histories must be a mapping")
        for port in PORTS:
            values = histories.get(port, [])
            if not isinstance(values, list):
                raise ValueError(f"history for {port} must be a list")
            for value in values:
                model.observe(port, float(value))
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LossModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
