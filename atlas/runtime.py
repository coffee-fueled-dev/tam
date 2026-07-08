"""Minimal self-organizing atlas runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atlas.core import AtlasDecision, ChartRecord, InferredSituation, PortView, ProjectedChart, RetrievalMatch
from atlas.encoder import SituationEncoder
from atlas.projector import ChartProjector
from atlas.retrieval import ChartRetriever
from atlas.store import AtlasStore

Tensor = torch.Tensor


@dataclass
class AtlasStep:
    """One atlas pass over an observed context."""

    inferred: InferredSituation
    matches: list[RetrievalMatch]
    projected_charts: list[ProjectedChart]
    chosen_port: PortView
    decision: AtlasDecision = "reuse"
    spawned_chart_id: str | None = None
    ambiguity_score: float = 0.0
    trusted_calibration_update: bool = False
    geometry_center_shift: float = 0.0
    geometry_radius_shift: float = 0.0


class AtlasRuntime:
    """Encode, retrieve, project, then reuse or spawn a chart."""

    def __init__(
        self,
        encoder: SituationEncoder,
        store: AtlasStore,
        retriever: ChartRetriever,
        projector: ChartProjector,
        top_k: int = 5,
        spawn_threshold: float = 1.0,
        spawn_rank: int = 2,
        default_radius: float = 0.5,
        retrieval_weight: float = 0.1,
        stability_weight: float = 0.15,
        support_weight: float = 0.25,
        ambiguity_margin: float = 0.15,
        reuse_tolerance: float = 0.35,
        trusted_support_threshold: float = 0.55,
        geometry_center_ema: float = 0.25,
        geometry_radius_ema: float = 0.25,
        geometry_radius_floor: float = 0.05,
    ):
        self.encoder = encoder
        self.store = store
        self.retriever = retriever
        self.projector = projector
        self.top_k = top_k
        self.spawn_threshold = spawn_threshold
        self.spawn_rank = spawn_rank
        self.default_radius = default_radius
        self.retrieval_weight = retrieval_weight
        self.stability_weight = stability_weight
        self.support_weight = support_weight
        self.ambiguity_margin = ambiguity_margin
        self.reuse_tolerance = reuse_tolerance
        self.trusted_support_threshold = trusted_support_threshold
        self.geometry_center_ema = geometry_center_ema
        self.geometry_radius_ema = geometry_radius_ema
        self.geometry_radius_floor = geometry_radius_floor
        self.step_count = 0

    def step(self, observed_context: Tensor, support_example: str | None = None) -> AtlasStep:
        inferred = self.encoder.infer(observed_context)
        matches = self.retriever.retrieve(inferred.query_key, self.store.all_charts(), top_k=self.top_k)
        projected = [
            self.projector.project(inferred, self.store.get_chart(match.chart_id))
            for match in matches
        ]
        chosen_projected, ambiguity_score, decision = self._choose_or_spawn(matches, inferred, projected)
        chosen_projected.port_view.decision = decision
        spawned_chart_id = None
        trusted_calibration_update = False
        geometry_center_shift = 0.0
        geometry_radius_shift = 0.0
        if chosen_projected.chart.chart_id not in {item.chart.chart_id for item in projected}:
            spawned_chart_id = chosen_projected.chart.chart_id
        else:
            residual = float(chosen_projected.raw_residual.item())
            self.store.observe_calibration(chosen_projected.chart.chart_id, residual=residual)
            if self._is_trusted_reuse(decision, chosen_projected):
                trusted_calibration_update = True
                chosen_projected.trusted_for_calibration = True
                chosen_projected.port_view.trusted_for_calibration = True
                self.store.update_calibration(chosen_projected.chart.chart_id, residual=residual)
                geometry_center_shift, geometry_radius_shift = self.store.adapt_chart_geometry(
                    chosen_projected.chart.chart_id,
                    chosen_projected.local_coords,
                    center_ema=self.geometry_center_ema,
                    radius_ema=self.geometry_radius_ema,
                    radius_floor=self.geometry_radius_floor,
                )

        self.store.update_stats(
            chosen_projected.chart.chart_id,
            support_score=float(chosen_projected.support_score.item()),
            contradiction=float(chosen_projected.contradiction.item()),
            support_example=support_example,
        )
        self.step_count += 1
        return AtlasStep(
            inferred=inferred,
            matches=matches,
            projected_charts=projected if projected else [chosen_projected],
            chosen_port=chosen_projected.port_view,
            decision=decision,
            spawned_chart_id=spawned_chart_id,
            ambiguity_score=ambiguity_score,
            trusted_calibration_update=trusted_calibration_update,
            geometry_center_shift=geometry_center_shift,
            geometry_radius_shift=geometry_radius_shift,
        )

    def _choose_or_spawn(
        self,
        matches: list[RetrievalMatch],
        inferred: InferredSituation,
        projected: list[ProjectedChart],
    ) -> tuple[ProjectedChart, float, AtlasDecision]:
        if not projected:
            return self._spawn_chart(inferred), 0.0, "spawn"

        match_by_chart = {match.chart_id: match for match in matches}
        scored: list[tuple[float, ProjectedChart, float, float]] = []
        for item in projected:
            match = match_by_chart.get(item.chart.chart_id)
            retrieval_prior = self.store.retrieval_strength(
                item.chart.chart_id,
                match.distance if match is not None else 1.0,
            )
            stability_prior = self.store.stability_prior(item.chart.chart_id)
            support_confidence = (
                float(item.support_confidence.item())
                if item.support_confidence is not None
                else float(item.support_score.item() > 0.0)
            )
            selection_score = (
                float(item.contradiction.item())
                - (self.support_weight * support_confidence)
                - (self.retrieval_weight * retrieval_prior)
                - (self.stability_weight * stability_prior)
            )
            item.selection_score = torch.tensor(selection_score, dtype=torch.float32)
            item.port_view.selection_score = item.selection_score
            scored.append((selection_score, item, retrieval_prior, stability_prior))

        scored.sort(key=lambda bundle: bundle[0])
        best_score, best, retrieval_prior, stability_prior = scored[0]
        if len(scored) > 1:
            second_score = scored[1][0]
            ambiguity_score = max(0.0, self.ambiguity_margin - (second_score - best_score))
        else:
            ambiguity_score = 0.0
        best.port_view.ambiguity = torch.tensor(ambiguity_score, dtype=torch.float32)

        should_spawn = not best.port_view.is_supported()
        if should_spawn:
            self.store.increment_spawn(best.chart.chart_id)
            return self._spawn_chart(inferred, parent_id=best.chart.chart_id), ambiguity_score, "spawn"
        decision: AtlasDecision = "ambiguous" if ambiguity_score > 0.0 else "reuse"
        return best, ambiguity_score, decision

    def _is_trusted_reuse(self, decision: AtlasDecision, projected: ProjectedChart) -> bool:
        if decision != "reuse":
            return False
        if not projected.port_view.is_supported():
            return False
        if projected.support_confidence is None:
            return True
        return float(projected.support_confidence.item()) >= self.trusted_support_threshold

    def _spawn_chart(self, inferred: InferredSituation, parent_id: str | None = None) -> ProjectedChart:
        latent = inferred.situation_latent
        latent_dim = int(latent.shape[-1])
        local_dim = min(self.spawn_rank, latent_dim)
        basis = torch.eye(latent_dim, dtype=torch.float32, device=latent.device)[:, :local_dim]
        local_center = latent[:local_dim].detach().clone()
        local_radius = torch.full((local_dim,), self.default_radius, dtype=torch.float32, device=latent.device)
        chart = self.store.create_chart(
            retrieval_key=inferred.query_key.detach().clone(),
            projection_basis=basis.detach().clone(),
            local_center=local_center,
            local_radius=local_radius,
            support_threshold=self.spawn_threshold,
            created_step=self.step_count,
            parent_id=parent_id,
        )
        return self.projector.project(inferred, chart)
