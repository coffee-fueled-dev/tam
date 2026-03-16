"""Structured benchmark worlds for TAM v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from v4.benchmark.observation import FeatureGroup, ObservationFrame, flatten_frame
from v4.core import BindingRecord, ContextAtom, ContextEpisode, Situation

Tensor = torch.Tensor


@dataclass
class DriftZone:
    """A local region that adds context-dependent drift."""

    start_x: float
    end_x: float
    drift: Tensor


@dataclass
class DoorRegion:
    """A bottleneck that narrows the admissible corridor width."""

    x_center: float
    half_width: float
    opening_radius: float


@dataclass
class CorridorWorldConfig:
    """Configuration for one structured corridor benchmark variant."""

    state_dim: int = 2
    corridor_length: float = 12.0
    corridor_half_width: float = 1.5
    forward_step: float = 0.6
    control_scale: float = 0.5
    noise_scale: float = 0.03
    episode_points: int = 4
    partial_observation_noise: float = 0.02
    include_energy: bool = True
    shuffle_features: bool = False
    include_drift_cues: bool = True
    hidden_mode: int = 0
    drift_zones: List[DriftZone] = field(default_factory=list)
    doors: List[DoorRegion] = field(default_factory=list)


class StructuredCorridorWorld:
    """A corridor world with repeated motifs and partial observability.

    State layout:
    - dimension 0: forward progress
    - remaining dimensions: transverse offsets
    """

    def __init__(self, config: CorridorWorldConfig):
        self.config = config
        self._energy = 1.0

    def initial_observed_state(self) -> Tensor:
        return torch.zeros(self.config.state_dim, dtype=torch.float32)

    def port_drifts(self) -> Dict[str, Tensor]:
        drifts: Dict[str, Tensor] = {
            "port_0": self._vector_with_dims(self.config.forward_step, [])
        }
        port_idx = 1
        for dim_idx in range(1, self.config.state_dim):
            positive = torch.zeros(self.config.state_dim, dtype=torch.float32)
            negative = torch.zeros(self.config.state_dim, dtype=torch.float32)
            positive[dim_idx] = self.config.control_scale
            negative[dim_idx] = -self.config.control_scale
            drifts[f"port_{port_idx}"] = positive
            drifts[f"port_{port_idx + 1}"] = negative
            port_idx += 2
        return drifts

    def observe(self, observed_state: Tensor) -> ContextAtom:
        frame = self._make_frame(observed_state)
        return ContextAtom(
            value=flatten_frame(frame),
            kind="structured_observation",
            metadata={
                "frame": frame,
                "physical_state": observed_state.clone(),
            },
        )

    def bind(self, binding: BindingRecord, situation: Situation) -> ContextEpisode:
        start = situation.observed_state
        drift_map = self.port_drifts()
        intended = drift_map.get(
            binding.selection.port_name,
            torch.zeros(self.config.state_dim, dtype=torch.float32, device=start.device),
        ).to(start.device)

        current = start.clone()
        atoms: List[ContextAtom] = [self.observe(current)]
        for step_idx in range(self.config.episode_points - 1):
            alpha = float(step_idx + 1) / float(self.config.episode_points - 1)
            target = start + alpha * intended
            current = self._apply_dynamics(current, target - current)
            atoms.append(self.observe(current))

        return ContextEpisode(
            atoms=atoms,
            metadata={
                "goal_reached": 1.0 if self._goal_reached(current) else 0.0,
                "hidden_mode": float(self.config.hidden_mode),
            },
        )

    def _vector_with_dims(self, forward_value: float, transverse_values: List[float]) -> Tensor:
        vector = torch.zeros(self.config.state_dim, dtype=torch.float32)
        vector[0] = forward_value
        for idx, value in enumerate(transverse_values, start=1):
            if idx < self.config.state_dim:
                vector[idx] = value
        return vector

    def _goal_reached(self, state: Tensor) -> bool:
        return bool(state[0].item() >= self.config.corridor_length)

    def _door_radius(self, x_position: float) -> float:
        radius = self.config.corridor_half_width
        for door in self.config.doors:
            if abs(x_position - door.x_center) <= door.half_width:
                radius = min(radius, door.opening_radius)
        return radius

    def _active_drift(self, state: Tensor) -> Tensor:
        drift = torch.zeros(self.config.state_dim, dtype=torch.float32, device=state.device)
        for zone in self.config.drift_zones:
            if zone.start_x <= float(state[0].item()) <= zone.end_x:
                drift = drift + zone.drift.to(state.device)
        if self.config.hidden_mode == 1 and self.config.state_dim > 1:
            hidden = torch.zeros_like(drift)
            hidden[1] = 0.1
            drift = drift + hidden
        return drift

    def _apply_dynamics(self, state: Tensor, control: Tensor) -> Tensor:
        noise = torch.randn_like(state) * self.config.noise_scale
        candidate = state + control + self._active_drift(state) + noise
        candidate[0] = torch.clamp(candidate[0], 0.0, self.config.corridor_length)

        door_radius = self._door_radius(float(candidate[0].item()))
        for dim_idx in range(1, self.config.state_dim):
            candidate[dim_idx] = torch.clamp(candidate[dim_idx], -door_radius, door_radius)

        self._energy = max(0.0, self._energy - 0.01 * float(torch.norm(control).item()))
        return candidate

    def _make_frame(self, state: Tensor) -> ObservationFrame:
        feature_groups: List[FeatureGroup] = []
        state_dim = self.config.state_dim

        forward_remaining = torch.tensor(
            [self.config.corridor_length - float(state[0].item())],
            dtype=torch.float32,
        )
        boundary_clearance = torch.full((state_dim,), self.config.corridor_half_width, dtype=torch.float32)
        boundary_clearance[0] = forward_remaining.item()
        door_clearance = torch.full((state_dim,), self._door_radius(float(state[0].item())), dtype=torch.float32)
        door_alignment = torch.zeros(state_dim, dtype=torch.float32)
        drift_cue = self._active_drift(state)
        local_energy = torch.tensor([self._energy], dtype=torch.float32)

        for dim_idx in range(1, state_dim):
            door_alignment[dim_idx] = door_clearance[dim_idx] - abs(float(state[dim_idx].item()))
            boundary_clearance[dim_idx] = self.config.corridor_half_width - abs(float(state[dim_idx].item()))

        feature_groups.append(FeatureGroup(name="boundary_clearance", values=boundary_clearance))
        feature_groups.append(FeatureGroup(name="door_alignment", values=door_alignment))
        if self.config.include_drift_cues:
            feature_groups.append(FeatureGroup(name="drift_cue", values=drift_cue))
        feature_groups.append(FeatureGroup(name="forward_remaining", values=forward_remaining, per_dimension=False))
        if self.config.include_energy:
            feature_groups.append(FeatureGroup(name="energy", values=local_energy, per_dimension=False))

        if self.config.shuffle_features:
            reordered: List[FeatureGroup] = []
            for index in range(len(feature_groups)):
                reordered.append(feature_groups[(index * 2) % len(feature_groups)])
            feature_groups = reordered

        noisy_state = state + torch.randn_like(state) * self.config.partial_observation_noise
        return ObservationFrame(
            state=noisy_state.to(dtype=torch.float32),
            feature_groups=feature_groups,
            metadata={
                "door_radius": float(self._door_radius(float(state[0].item()))),
                "hidden_mode": float(self.config.hidden_mode),
                "energy": self._energy,
            },
        )
