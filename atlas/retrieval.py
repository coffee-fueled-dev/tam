"""Top-k chart retrieval for the atlas runtime."""

from __future__ import annotations

from typing import Iterable, Literal

import torch

from atlas.core import ChartRecord, RetrievalMatch, ensure_1d

Tensor = torch.Tensor
Metric = Literal["cosine", "l2"]


def _cosine_score(query_key: Tensor, chart_key: Tensor) -> float:
    query = ensure_1d(query_key)
    key = ensure_1d(chart_key)
    query_norm = torch.norm(query)
    key_norm = torch.norm(key)
    if query_norm.item() <= 1e-8 or key_norm.item() <= 1e-8:
        return 0.0
    return float(torch.dot(query, key).item() / (query_norm.item() * key_norm.item()))


def _l2_distance(query_key: Tensor, chart_key: Tensor) -> float:
    query = ensure_1d(query_key)
    key = ensure_1d(chart_key)
    return float(torch.norm(query - key).item())


class ChartRetriever:
    """Simple brute-force top-k retrieval over chart keys."""

    def __init__(self, metric: Metric = "cosine"):
        self.metric = metric

    def retrieve(
        self,
        query_key: Tensor,
        charts: Iterable[ChartRecord],
        top_k: int = 5,
    ) -> list[RetrievalMatch]:
        matches: list[RetrievalMatch] = []
        for chart in charts:
            if self.metric == "l2":
                distance = _l2_distance(query_key, chart.retrieval_key)
                score = -distance
            else:
                score = _cosine_score(query_key, chart.retrieval_key)
                distance = 1.0 - score
            matches.append(
                RetrievalMatch(
                    chart_id=chart.chart_id,
                    distance=distance,
                    score=score,
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[: max(top_k, 0)]
