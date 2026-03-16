"""Minimal example for the v4 reference runtime."""

from __future__ import annotations

import torch

from v4.runtime_torch import TorchAgent, initial_history
from v4.training import train_agent
from v4.worlds import PlaneWorld


def build_demo_world() -> PlaneWorld:
    return PlaneWorld(
        drift_by_port={
            "port_0": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "port_1": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "port_2": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
        noise_scale=0.02,
    )


def main() -> None:
    torch.manual_seed(7)

    world = build_demo_world()
    agent = TorchAgent(observed_dim=2, latent_dim=4, n_ports=3, fiber_dim=1)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)

    trace = train_agent(agent, world, optimizer=optimizer, steps=50)
    print(f"Initial contradiction: {trace.steps[0].contradiction:.4f}")
    print(f"Final contradiction:   {trace.steps[-1].contradiction:.4f}")

    observed_state = world.initial_observed_state()
    history = initial_history()
    outcome, _ = agent.run_cycle(
        world=world,
        history=history,
        step=0,
        observed_state=observed_state,
        sample=False,
    )
    print(f"Selected port:         {outcome.binding.selection.port_name}")
    print(f"Success:               {outcome.success}")
    print(f"Contradiction:         {outcome.contradiction.item():.4f}")
    print(f"Realized delta:        {outcome.realized_transition.delta.tolist()}")


if __name__ == "__main__":
    main()
