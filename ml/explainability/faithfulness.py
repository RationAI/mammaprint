"""Patch-flipping faithfulness evaluation for tile explanations."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ml.explainability.attribution import (
    _attention_components,
    _evaluation_mode,
    _forward_raw_unchecked,
    _head_in_chunks,
    _validate_baseline,
    _validate_head_batch_size,
    validate_pipeline,
)
from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.base import Aggregator
from ml.models.aggregators.max import MaxPool
from ml.models.aggregators.mean import MeanPool
from ml.models.heads.base import Head


@dataclass(frozen=True)
class PatchFlippingResult:
    """Target-specific patch-flipping curves and their raw-output AUCs.

    Shapes:
        - ``fractions``: ``(points,)``
        - ``descending`` / ``ascending``: ``(outputs, points)``
        - ``random``: ``(repeats, outputs, points)``
        - ``descending_auc`` / ``ascending_auc`` / ``srg``: ``(outputs,)``
        - ``random_auc``: ``(repeats, outputs)``

    For output ``o``, the descending and ascending curves use the ranking from
    ``tile_scores[:, o]`` and retain only model output ``o``.  SRG is the area
    between the curves: ``ascending_auc - descending_auc`` (higher is better).
    Because the integration interval is ``[0, 1]``, each AUC is also the
    trapezoid-weighted curve mean; it remains in the raw output's units and is
    deliberately not normalized across different targets.
    """

    fractions: Tensor
    descending: Tensor
    ascending: Tensor
    random: Tensor
    descending_auc: Tensor
    ascending_auc: Tensor
    random_auc: Tensor
    srg: Tensor


@dataclass(frozen=True)
class _AttentionState:
    normalized: Tensor
    logits: Tensor


def patch_flipping(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
    tile_scores: Tensor,
    baseline: Tensor,
    *,
    percentage_step: int = 1,
    random_repeats: int = 5,
    head_batch_size: int = 4096,
    generator: torch.Generator | None = None,
) -> PatchFlippingResult:
    """Evaluate a tile heatmap by progressively removing ranked tiles.

    The ranking is target-specific and the plotted values are raw outputs.  At
    100% removal, the configured neutral baseline embedding is evaluated as a
    one-tile bag, ensuring that every curve has a defined endpoint.

    The implementation is exact but avoids repeatedly running the aggregator.
    Each removal order is reduced once using suffix sufficient statistics:
    sums for mean, maxima for max, and precomputed normalized features and gated
    attention logits for attention.  Only the small collection of pooled vectors
    is sent through the complete prediction head.
    """
    validate_pipeline(aggregator, head, bag)
    baseline = _validate_baseline(baseline, bag)
    _validate_tile_scores(tile_scores, bag, head)
    _validate_head_batch_size(head_batch_size)
    if not 1 <= percentage_step <= 100:
        raise ValueError(
            "percentage_step must be between 1 and 100; "
            f"got {percentage_step}."
        )
    if random_repeats < 1:
        raise ValueError(
            f"random_repeats must be at least one; got {random_repeats}."
        )

    percentages = list(range(0, 100, percentage_step))
    if percentages[-1] != 100:
        percentages.append(100)
    fractions = torch.tensor(
        percentages,
        dtype=bag.dtype,
        device=bag.device,
    ).div_(100)
    removal_counts = [bag.shape[0] * percentage // 100 for percentage in percentages]
    removal_counts[-1] = bag.shape[0]

    with _evaluation_mode(aggregator, head), torch.no_grad():
        attention_state = _prepare_attention_state(aggregator, bag)
        baseline_output = _forward_raw_unchecked(
            aggregator, head, baseline.unsqueeze(0)
        ).scores
        full_output = _forward_raw_unchecked(aggregator, head, bag).scores

        descending = torch.empty(
            (head.out_dim, len(percentages)), dtype=bag.dtype, device=bag.device
        )
        ascending = torch.empty_like(descending)

        for output_index in range(head.out_dim):
            descending_order = torch.argsort(
                tile_scores[:, output_index], descending=True, stable=True
            )
            ascending_order = torch.argsort(
                tile_scores[:, output_index], descending=False, stable=True
            )
            descending_outputs = _ordered_subset_outputs(
                aggregator,
                head,
                bag,
                descending_order,
                removal_counts,
                baseline_output,
                full_output,
                head_batch_size,
                attention_state,
            )
            ascending_outputs = _ordered_subset_outputs(
                aggregator,
                head,
                bag,
                ascending_order,
                removal_counts,
                baseline_output,
                full_output,
                head_batch_size,
                attention_state,
            )
            descending[output_index] = descending_outputs[:, output_index]
            ascending[output_index] = ascending_outputs[:, output_index]

        random_curves = torch.empty(
            (random_repeats, head.out_dim, len(percentages)),
            dtype=bag.dtype,
            device=bag.device,
        )
        for repeat in range(random_repeats):
            random_order = torch.randperm(bag.shape[0], generator=generator).to(
                bag.device
            )
            random_outputs = _ordered_subset_outputs(
                aggregator,
                head,
                bag,
                random_order,
                removal_counts,
                baseline_output,
                full_output,
                head_batch_size,
                attention_state,
            )
            random_curves[repeat] = random_outputs.transpose(0, 1)

    descending_auc = torch.trapezoid(descending, fractions, dim=1)
    ascending_auc = torch.trapezoid(ascending, fractions, dim=1)
    random_auc = torch.trapezoid(random_curves, fractions, dim=2)
    return PatchFlippingResult(
        fractions=fractions,
        descending=descending,
        ascending=ascending,
        random=random_curves,
        descending_auc=descending_auc,
        ascending_auc=ascending_auc,
        random_auc=random_auc,
        srg=ascending_auc - descending_auc,
    )


def _ordered_subset_outputs(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
    removal_order: Tensor,
    removal_counts: list[int],
    baseline_output: Tensor,
    full_output: Tensor,
    head_batch_size: int,
    attention_state: _AttentionState | None,
) -> Tensor:
    num_tiles = bag.shape[0]
    checkpoints = sorted(
        {count for count in removal_counts if 0 < count < num_tiles}, reverse=True
    )
    pooled_by_count: dict[int, Tensor] = {}

    if checkpoints:
        if isinstance(aggregator, MeanPool):
            pooled_by_count = _mean_suffix_pooling(bag, removal_order, checkpoints)
        elif isinstance(aggregator, MaxPool):
            pooled_by_count = _max_suffix_pooling(bag, removal_order, checkpoints)
        else:
            if attention_state is None:
                raise RuntimeError("Missing precomputed gated-attention state.")
            pooled_by_count = _attention_suffix_pooling(
                attention_state,
                removal_order,
                checkpoints,
            )

    unique_intermediate = sorted(pooled_by_count)
    predictions_by_count: dict[int, Tensor] = {
        0: full_output,
        num_tiles: baseline_output,
    }
    if unique_intermediate:
        pooled = torch.stack(
            [pooled_by_count[count] for count in unique_intermediate], dim=0
        )
        predictions = _head_in_chunks(head, pooled, head_batch_size)
        predictions_by_count.update(
            zip(unique_intermediate, predictions.unbind(dim=0), strict=True)
        )

    return torch.stack(
        [predictions_by_count[count] for count in removal_counts], dim=0
    )


def _mean_suffix_pooling(
    bag: Tensor,
    order: Tensor,
    checkpoints: list[int],
) -> dict[int, Tensor]:
    ordered = bag.index_select(0, order)
    running_sum = torch.zeros(bag.shape[1], dtype=bag.dtype, device=bag.device)
    previous = bag.shape[0]
    result: dict[int, Tensor] = {}
    for count in checkpoints:
        running_sum = running_sum + ordered[count:previous].sum(dim=0)
        result[count] = running_sum / (bag.shape[0] - count)
        previous = count
    return result


def _max_suffix_pooling(
    bag: Tensor,
    order: Tensor,
    checkpoints: list[int],
) -> dict[int, Tensor]:
    ordered = bag.index_select(0, order)
    running_maximum: Tensor | None = None
    previous = bag.shape[0]
    result: dict[int, Tensor] = {}
    for count in checkpoints:
        segment_maximum = ordered[count:previous].amax(dim=0)
        running_maximum = (
            segment_maximum
            if running_maximum is None
            else torch.maximum(running_maximum, segment_maximum)
        )
        result[count] = running_maximum
        previous = count
    return result


def _attention_suffix_pooling(
    state: _AttentionState,
    order: Tensor,
    checkpoints: list[int],
) -> dict[int, Tensor]:
    normalized = state.normalized.index_select(0, order)
    logits = state.logits.index_select(0, order)
    running_maximum: Tensor | None = None
    running_denominator: Tensor | None = None
    running_numerator: Tensor | None = None
    previous = normalized.shape[0]
    result: dict[int, Tensor] = {}

    for count in checkpoints:
        segment_logits = logits[count:previous]
        segment_features = normalized[count:previous]
        segment_maximum = segment_logits.amax()
        segment_weights = torch.exp(segment_logits - segment_maximum)
        segment_denominator = segment_weights.sum()
        segment_numerator = (
            segment_weights.unsqueeze(1) * segment_features
        ).sum(dim=0)

        if running_maximum is None:
            running_maximum = segment_maximum
            running_denominator = segment_denominator
            running_numerator = segment_numerator
        else:
            if running_denominator is None or running_numerator is None:
                raise RuntimeError("Incomplete gated-attention suffix state.")
            combined_maximum = torch.maximum(running_maximum, segment_maximum)
            old_scale = torch.exp(running_maximum - combined_maximum)
            segment_scale = torch.exp(segment_maximum - combined_maximum)
            running_denominator = (
                running_denominator * old_scale
                + segment_denominator * segment_scale
            )
            running_numerator = (
                running_numerator * old_scale
                + segment_numerator * segment_scale
            )
            running_maximum = combined_maximum

        if running_denominator is None or running_numerator is None:
            raise RuntimeError("Failed to construct gated-attention suffix state.")
        result[count] = running_numerator / running_denominator
        previous = count
    return result


def _prepare_attention_state(
    aggregator: Aggregator[Tensor],
    bag: Tensor,
) -> _AttentionState | None:
    if not isinstance(aggregator, AttentionMIL):
        return None
    normalized, logits, _ = _attention_components(aggregator, bag)
    return _AttentionState(normalized=normalized, logits=logits)


def _validate_tile_scores(tile_scores: Tensor, bag: Tensor, head: Head) -> None:
    expected = (bag.shape[0], head.out_dim)
    if tile_scores.shape != expected:
        raise ValueError(
            f"tile_scores must have shape {expected}; got {tuple(tile_scores.shape)}."
        )
    if tile_scores.device != bag.device:
        raise ValueError(
            f"tile_scores are on {tile_scores.device}, but the bag is on {bag.device}."
        )
    if not tile_scores.is_floating_point():
        raise ValueError("tile_scores must be floating point.")
    if not bool(torch.isfinite(tile_scores).all()):
        raise ValueError("tile_scores contain NaN or infinite values.")


__all__ = ["PatchFlippingResult", "patch_flipping"]
