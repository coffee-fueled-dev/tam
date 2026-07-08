"""Tiny end-to-end example for the atlas runtime."""

from __future__ import annotations

from atlas.encoder import IdentitySituationEncoder
from atlas.projector import ChartProjector
from atlas.retrieval import ChartRetriever
from atlas.runtime import AtlasRuntime
from atlas.store import AtlasStore
from atlas.worlds import default_world


def main() -> None:
    runtime = AtlasRuntime(
        encoder=IdentitySituationEncoder(),
        store=AtlasStore(),
        retriever=ChartRetriever(metric="l2"),
        projector=ChartProjector(default_threshold=1.0),
        top_k=3,
        spawn_threshold=1.0,
        spawn_rank=2,
        default_radius=0.35,
    )
    world = default_world()

    for step_idx, observation in enumerate(world.sequence()):
        result = runtime.step(observation, support_example=f"obs_{step_idx}")
        spawn_text = result.spawned_chart_id if result.spawned_chart_id is not None else "reuse"
        print(
            f"step={step_idx} port={result.chosen_port.port_name} "
            f"contradiction={result.chosen_port.contradiction.item():.3f} "
            f"support={result.chosen_port.support_score.item():.3f} "
            f"event={spawn_text} charts={len(runtime.store)}"
        )


if __name__ == "__main__":
    main()
