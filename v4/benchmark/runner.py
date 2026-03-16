"""Runner for the v4 benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from v4.benchmark.metrics import (
    BenchmarkSummary,
    summarize_geometry,
    summarize_representation,
    summarize_ports,
    summarize_task,
)
from v4.benchmark.observation import dimension_feature_matrix, feature_order
from v4.benchmark.runtime import (
    BenchmarkGeometryHead,
    DimensionSpecificFeatureEncoder,
    EncoderMode,
    GeometryMode,
    RawObservationEncoder,
    SharedFeatureEncoder,
)
from v4.benchmark.worlds import CorridorWorldConfig, StructuredCorridorWorld
from v4.core import (
    BindingOutcome,
    BindingRecord,
    ClaimedRegion,
    ContextWindow,
    LatentTransition,
    PortFiber,
    PortSelection,
    Situation,
)
from v4.theory import next_situation
from v4.world import WorldHistory
from v4.benchmark.worlds import DoorRegion, DriftZone

Tensor = torch.Tensor


@dataclass
class BenchmarkVariant:
    name: str
    world_config: CorridorWorldConfig


@dataclass
class BenchmarkConfig:
    name: str
    encoder_mode: EncoderMode
    geometry_mode: GeometryMode
    steps: int = 80
    learning_rate: float = 1e-2


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    variant: BenchmarkVariant
    summary: BenchmarkSummary


class BenchmarkAgent(nn.Module):
    """A small benchmark-only agent with configurable observation path."""

    def __init__(
        self,
        observed_dim: int,
        latent_dim: int,
        n_ports: int,
        state_dim: int,
        per_dim_feature_dim: int,
        config: BenchmarkConfig,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.config = config
        if config.encoder_mode == "raw":
            self.encoder = RawObservationEncoder(observed_dim, latent_dim)
        else:
            if config.encoder_mode == "shared":
                self.encoder = SharedFeatureEncoder(
                    observed_dim=observed_dim,
                    latent_dim=latent_dim,
                    per_dim_feature_dim=per_dim_feature_dim,
                )
            else:
                self.encoder = DimensionSpecificFeatureEncoder(
                    observed_dim=observed_dim,
                    latent_dim=latent_dim,
                    state_dim=state_dim,
                    per_dim_feature_dim=per_dim_feature_dim,
                )
        self.geometry = BenchmarkGeometryHead(
            latent_dim=latent_dim,
            n_ports=n_ports,
            geometry_mode=config.geometry_mode,
            fiber_dim=1,
        )
        self.interpreter = nn.Sequential(
            nn.Linear(observed_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

    def encode(self, atom) -> tuple[Tensor, Tensor | None]:
        encoded = self.encoder(atom)
        return encoded.encoded, encoded.feature_matrix

    def select_binding(self, situation: Situation) -> BindingRecord:
        anchors, bases, radii, confidences, logits = self.geometry(situation.latent_state)
        probabilities = torch.softmax(logits, dim=-1)
        port_index = torch.distributions.Categorical(probabilities).sample()
        chosen = int(port_index.item())
        fiber = PortFiber(
            anchor=anchors[chosen],
            basis=bases[chosen],
            radius=radii[chosen],
            confidence=confidences[chosen],
        )
        selection = PortSelection(
            port_name=f"port_{chosen}",
            port_index=chosen,
            logit=logits[chosen],
            probability=probabilities[chosen],
            log_probability=torch.log(probabilities[chosen] + 1e-6),
        )
        return BindingRecord(
            situation_step=situation.step,
            selection=selection,
            claimed_region=ClaimedRegion(
                port_name=selection.port_name,
                fiber=fiber,
                bind_context=situation.prior_context.copy(),
            ),
        )

    def interpret_episode(self, episode_atom_value: Tensor, start_latent_state: Tensor) -> Tensor:
        return self.interpreter(episode_atom_value) - start_latent_state


def _make_variants() -> List[BenchmarkVariant]:
    base_doors = [
        {"x_center": 4.0, "half_width": 0.5, "opening_radius": 0.6},
        {"x_center": 8.0, "half_width": 0.7, "opening_radius": 0.4},
    ]
    base_drift = [
        {"start_x": 2.0, "end_x": 3.2, "drift": torch.tensor([0.0, 0.15])},
        {"start_x": 6.0, "end_x": 7.0, "drift": torch.tensor([0.0, -0.15])},
    ]

    def make_config(state_dim: int, hidden_mode: int = 0, shuffle_features: bool = False) -> CorridorWorldConfig:
        drift_zones = []
        for zone in base_drift:
            drift = torch.zeros(state_dim, dtype=torch.float32)
            drift[: min(len(zone["drift"]), state_dim)] = zone["drift"][: min(len(zone["drift"]), state_dim)]
            drift_zones.append(
                DriftZone(
                    start_x=zone["start_x"],
                    end_x=zone["end_x"],
                    drift=drift,
                )
            )
        doors = [
            DoorRegion(
                x_center=door["x_center"],
                half_width=door["half_width"],
                opening_radius=door["opening_radius"],
            )
            for door in base_doors
        ]
        return CorridorWorldConfig(
            state_dim=state_dim,
            hidden_mode=hidden_mode,
            shuffle_features=shuffle_features,
            drift_zones=drift_zones,
            doors=doors,
        )

    return [
        BenchmarkVariant(name="corridor_2d", world_config=make_config(2)),
        BenchmarkVariant(name="corridor_2d_hidden", world_config=make_config(2, hidden_mode=1)),
        BenchmarkVariant(name="corridor_4d", world_config=make_config(4)),
        BenchmarkVariant(name="corridor_4d_shuffled", world_config=make_config(4, shuffle_features=True)),
    ]


def default_ablation_configs() -> List[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            name="shared_fiber",
            encoder_mode="shared",
            geometry_mode="fiber",
        ),
        BenchmarkConfig(
            name="raw_fiber",
            encoder_mode="raw",
            geometry_mode="fiber",
        ),
        BenchmarkConfig(
            name="shared_diagonal",
            encoder_mode="shared",
            geometry_mode="diagonal",
        ),
        BenchmarkConfig(
            name="shared_point",
            encoder_mode="shared",
            geometry_mode="point",
        ),
        BenchmarkConfig(
            name="dimension_specific_fiber",
            encoder_mode="dimension_specific",
            geometry_mode="fiber",
        ),
    ]


def _policy_weighted_loss(outcome: BindingOutcome) -> Tensor:
    contradiction = outcome.contradiction
    log_probability = outcome.binding.selection.log_probability
    if log_probability is None:
        return contradiction
    return contradiction + 0.1 * contradiction.detach() * (-log_probability)


def _run_one(
    config: BenchmarkConfig,
    variant: BenchmarkVariant,
) -> BenchmarkResult:
    world = StructuredCorridorWorld(variant.world_config)
    initial_atom = world.observe(world.initial_observed_state())
    observed_dim = int(initial_atom.value.shape[-1])
    initial_feature_matrix, _ = dimension_feature_matrix(initial_atom.metadata["frame"])
    n_ports = 1 + 2 * max(variant.world_config.state_dim - 1, 0)
    agent = BenchmarkAgent(
        observed_dim=observed_dim,
        latent_dim=16,
        n_ports=n_ports,
        state_dim=variant.world_config.state_dim,
        per_dim_feature_dim=int(initial_feature_matrix.shape[-1]),
        config=config,
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    history = WorldHistory()
    physical_state = world.initial_observed_state()

    contradictions: List[float] = []
    radii: List[float] = []
    claim_successes: List[bool] = []
    task_successes: List[bool] = []
    confidences: List[float] = []
    selected_ports: List[str] = []
    progress_values: List[float] = []
    port_centers: Dict[str, List[Tensor]] = {}
    feature_matrices: List[Tensor] = []
    feature_orders: List[List[str]] = []

    for step in range(config.steps):
        optimizer.zero_grad()

        observation_atom = world.observe(physical_state)
        history.append_observation(observation_atom)
        encoded_latent, feature_matrix = agent.encode(observation_atom)
        if feature_matrix is not None:
            feature_matrices.append(feature_matrix.detach())
        frame = observation_atom.metadata["frame"]
        feature_orders.append(feature_order(frame))

        situation = Situation(
            step=step,
            latent_state=encoded_latent,
            observed_state=physical_state,
            prior_context=ContextWindow(atoms=list(history.atoms)),
        )

        binding = agent.select_binding(situation)
        episode = world.bind(binding, situation)
        history.append_episode(episode)

        next_atom = episode.atoms[-1]
        end_latent, feature_matrix_end = agent.encode(next_atom)
        if feature_matrix_end is not None:
            feature_matrices.append(feature_matrix_end.detach())
        frame = next_atom.metadata["frame"]
        feature_orders.append(feature_order(frame))

        realized_delta = end_latent - encoded_latent
        contradiction = binding.claimed_region.fiber.contradiction(realized_delta)
        outcome = BindingOutcome(
            binding=binding,
            episode=episode,
            realized_transition=LatentTransition(
                start=encoded_latent,
                end=end_latent,
            ),
            contradiction=contradiction,
            success=binding.claimed_region.fiber.contains(realized_delta),
            next_situation=next_situation(
                previous=situation,
                next_latent_state=end_latent,
                next_observed_state=next_atom.metadata["physical_state"],
                next_prior_context=ContextWindow(atoms=list(history.atoms)),
            ),
        )

        loss = _policy_weighted_loss(outcome)
        loss.backward()
        optimizer.step()

        contradictions.append(float(outcome.contradiction.item()))
        radii.append(float(torch.mean(binding.claimed_region.fiber.radius).item()))
        claim_successes.append(outcome.success)
        task_successes.append(bool(episode.metadata.get("goal_reached", 0.0) >= 1.0))
        confidences.append(float(binding.claimed_region.fiber.confidence.item()))
        selected_ports.append(binding.selection.port_name)
        progress_values.append(float(next_atom.metadata["physical_state"][0].item()))
        port_centers.setdefault(binding.selection.port_name, []).append(
            binding.claimed_region.fiber.anchor.detach()
        )

        physical_state = next_atom.metadata["physical_state"].detach()

    summary = BenchmarkSummary(
        geometry=summarize_geometry(contradictions, radii, claim_successes, confidences),
        ports=summarize_ports(selected_ports, port_centers),
        task=summarize_task(
            task_success_flags=task_successes,
            claim_success_flags=claim_successes,
            progresses=progress_values,
        ),
        representation=summarize_representation(feature_matrices, feature_orders),
        ablation_name=config.name,
        variant_name=variant.name,
    )
    return BenchmarkResult(config=config, variant=variant, summary=summary)


def run_benchmark_suite(
    configs: List[BenchmarkConfig] | None = None,
    variants: List[BenchmarkVariant] | None = None,
) -> List[BenchmarkResult]:
    """Run the benchmark suite across variants and ablations."""
    configs = configs if configs is not None else default_ablation_configs()
    variants = variants if variants is not None else _make_variants()

    results: List[BenchmarkResult] = []
    for variant in variants:
        for config in configs:
            torch.manual_seed(7)
            results.append(_run_one(config, variant))
    return results
