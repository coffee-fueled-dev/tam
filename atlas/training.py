"""Minimal training helpers for learned atlas encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn.functional as F

from atlas.encoder import ReferenceSituationEncoder, normalize_observed_context

if TYPE_CHECKING:
    from atlas.benchmark.worlds import ObservationEvent

Tensor = torch.Tensor


def _prototype_means(embeddings: Tensor, labels: Tensor, n_classes: int) -> Tensor:
    prototypes = []
    for class_idx in range(n_classes):
        mask = labels == class_idx
        prototypes.append(embeddings[mask].mean(dim=0))
    return torch.stack(prototypes, dim=0)


def train_reference_encoder_on_events(
    encoder: ReferenceSituationEncoder,
    events: Iterable["ObservationEvent"],
    *,
    seed: int = 7,
    steps: int = 80,
    learning_rate: float = 1e-2,
    margin: float = 1.0,
) -> ReferenceSituationEncoder:
    """Fit the learned encoder to separate regimes in a deterministic trace."""
    event_list = list(events)
    if not event_list:
        return encoder

    torch.manual_seed(seed)
    regime_to_index: dict[str, int] = {}
    observations = []
    label_values = []
    for event in event_list:
        regime_to_index.setdefault(event.regime_name, len(regime_to_index))
        observations.append(normalize_observed_context(event.observation))
        label_values.append(regime_to_index[event.regime_name])

    batch = torch.stack(observations, dim=0)
    labels = torch.tensor(label_values, dtype=torch.long)
    n_classes = len(regime_to_index)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    encoder.train()
    for _ in range(steps):
        optimizer.zero_grad()
        shared = encoder.shared(encoder.feature_dropout(batch))
        latents = encoder.latent_head(shared)
        queries = encoder.query_head(shared)

        latent_prototypes = _prototype_means(latents, labels, n_classes)
        query_prototypes = _prototype_means(queries, labels, n_classes)

        latent_logits = -torch.cdist(latents, latent_prototypes)
        query_logits = -torch.cdist(queries, query_prototypes)
        latent_ce = F.cross_entropy(latent_logits, labels)
        query_ce = F.cross_entropy(query_logits, labels)

        chosen_latent = latent_prototypes[labels]
        chosen_query = query_prototypes[labels]
        within_loss = F.mse_loss(latents, chosen_latent) + F.mse_loss(queries, chosen_query)

        prototype_dists = torch.cdist(query_prototypes, query_prototypes)
        separation_mask = ~torch.eye(n_classes, dtype=torch.bool)
        separation = prototype_dists[separation_mask]
        margin_loss = (
            torch.relu(margin - separation).pow(2).mean()
            if separation.numel()
            else torch.tensor(0.0, device=batch.device)
        )

        loss = latent_ce + query_ce + (0.5 * within_loss) + (0.2 * margin_loss)
        loss.backward()
        optimizer.step()

    encoder.eval()
    return encoder
