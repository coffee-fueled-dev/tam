"""Stage 1E categorical predictors with rational immutable commitments."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

OUTCOMES = (-1, 0, 1)
PORTS = ("a", "b")


@dataclass(frozen=True)
class Commitment:
    probabilities: Mapping[int, Fraction]
    cone: tuple[int, ...]
    cone_mass: Fraction

    def probability_pairs(self) -> dict[str, list[int]]:
        return {
            str(outcome): [
                self.probabilities[outcome].numerator,
                self.probabilities[outcome].denominator,
            ]
            for outcome in OUTCOMES
        }


def _dirichlet_probabilities(
    history: Sequence[int],
    alpha: Fraction,
) -> dict[int, Fraction]:
    counts = Counter(history)
    denominator = Fraction(len(history)) + alpha * len(OUTCOMES)
    return {
        outcome: (Fraction(counts[outcome]) + alpha) / denominator
        for outcome in OUTCOMES
    }


def cone_90(
    probabilities: Mapping[int, Fraction],
    mass: Fraction = Fraction(9, 10),
) -> tuple[tuple[int, ...], Fraction]:
    ranked = sorted(
        OUTCOMES,
        key=lambda outcome: (-probabilities[outcome], outcome),
    )
    cone: list[int] = []
    total = Fraction(0)
    for outcome in ranked:
        cone.append(outcome)
        total += probabilities[outcome]
        if total >= mass:
            break
    return tuple(cone), total


def build_commitment(
    probabilities: Mapping[int, Fraction],
    full_domain: bool = False,
    mass: Fraction = Fraction(9, 10),
) -> Commitment:
    probs = {outcome: Fraction(probabilities[outcome]) for outcome in OUTCOMES}
    if full_domain:
        cone = OUTCOMES
        cone_mass = sum((probs[outcome] for outcome in OUTCOMES), Fraction(0))
    else:
        cone, cone_mass = cone_90(probs, mass)
    return Commitment(MappingProxyType(probs), cone, cone_mass)


def brier_score(
    probabilities: Mapping[int, Fraction],
    outcome: int,
) -> Fraction:
    return sum(
        (probabilities[value] - (Fraction(1) if value == outcome else Fraction(0))) ** 2
        for value in OUTCOMES
    )


def negative_log_likelihood(
    probabilities: Mapping[int, Fraction],
    outcome: int,
) -> float:
    probability = float(probabilities[outcome])
    if probability <= 0.0:
        return 1e12
    import math

    return -math.log(probability)


def oracle_probabilities(regime: Mapping[str, Mapping[str, float]], port: str) -> dict[int, Fraction]:
    table = regime[port]
    return {
        outcome: Fraction(table[str(outcome)]).limit_denominator(100)
        for outcome in OUTCOMES
    }


class OutcomeModel:
    """Online categorical model with optional bounded history."""

    def __init__(
        self,
        history_size: int | None = 64,
        alpha: Fraction = Fraction(1, 2),
    ) -> None:
        self.history_size = history_size
        self.alpha = Fraction(alpha)
        self._histories: dict[str, deque[int] | list[int]] = {
            port: (
                deque(maxlen=history_size)
                if history_size is not None
                else []
            )
            for port in PORTS
        }
        self.frozen = False

    def probabilities(self, port: str) -> dict[int, Fraction]:
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        return _dirichlet_probabilities(self._histories[port], self.alpha)

    def predict(self, port: str, full_domain: bool = False) -> Commitment:
        return build_commitment(self.probabilities(port), full_domain=full_domain)

    def observe(self, port: str, outcome: int) -> None:
        if self.frozen:
            return
        if port not in PORTS:
            raise ValueError(f"unknown port: {port}")
        if outcome not in OUTCOMES:
            raise ValueError(f"unsupported outcome: {outcome}")
        self._histories[port].append(outcome)

    def freeze(self) -> None:
        self.frozen = True

    def clear(self) -> None:
        for history in self._histories.values():
            history.clear()
        self.frozen = False

    def state_payload(self) -> dict[str, object]:
        return {
            "alpha": [self.alpha.numerator, self.alpha.denominator],
            "frozen": self.frozen,
            "history_size": self.history_size,
            "histories": {
                port: list(self._histories[port]) for port in PORTS
            },
        }

    def to_dict(self) -> dict[str, object]:
        return self.state_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OutcomeModel":
        alpha_pair = data["alpha"]
        assert isinstance(alpha_pair, list)
        model = cls(
            history_size=(
                None if data["history_size"] is None else int(data["history_size"])
            ),
            alpha=Fraction(int(alpha_pair[0]), int(alpha_pair[1])),
        )
        histories = data["histories"]
        assert isinstance(histories, Mapping)
        for port in PORTS:
            for value in histories[port]:
                model.observe(port, int(value))
        if bool(data.get("frozen", False)):
            model.freeze()
        return model


def make_model(variant: str) -> OutcomeModel:
    if variant in ("window64", "full_domain", "frozen200"):
        return OutcomeModel(history_size=64)
    if variant == "cumulative":
        return OutcomeModel(history_size=None)
    raise ValueError(f"unknown variant: {variant}")


def total_variation(
    left: Iterable[int],
    right: Iterable[int],
) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    left_n = sum(left_counts.values()) or 1
    right_n = sum(right_counts.values()) or 1
    return 0.5 * sum(
        abs(left_counts[outcome] / left_n - right_counts[outcome] / right_n)
        for outcome in OUTCOMES
    )
