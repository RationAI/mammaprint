"""Post-hoc tile attribution for embedding-based MIL models.

The functions in this module operate on an already encoded bag and deliberately
call only the configured aggregator and prediction head.  They therefore explain
the raw model output (logit or regression value), without changing or retraining
the model and without invoking the embedding encoder.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.base import Aggregator
from ml.models.aggregators.max import MaxPool
from ml.models.aggregators.mean import MeanPool
from ml.models.heads.base import Head
from ml.models.heads.linear import LinearHead
from ml.models.heads.mlp import MLPHead


SupportedAggregator = MeanPool | MaxPool | AttentionMIL
SupportedHead = LinearHead | MLPHead
_MAX_IG_TILE_BATCH_SIZE = 4096


@dataclass(frozen=True)
class RawPipelineOutput:
    """Raw output of the aggregator and head for one bag."""

    scores: Tensor
    pooled: Tensor
    attention: Tensor | None


@dataclass(frozen=True)
class LeaveOneOutResult:
    """Exact leave-one-tile-out predictions and necessity scores.

    ``delta[i, o]`` is ``full_output[o] - without_tile[i, o]``.  Positive values
    therefore mean that tile ``i`` raises output ``o`` in the context of the
    complete bag.
    """

    full_output: Tensor
    without_tile: Tensor
    delta: Tensor


@dataclass(frozen=True)
class IntegratedGradientsResult:
    """Signed, tile-summed Integrated Gradients for every raw output."""

    attributions: Tensor
    full_output: Tensor
    baseline_output: Tensor
    completeness_residual: Tensor
    relative_completeness_error: Tensor


@dataclass(frozen=True)
class BagAttributionResult:
    """All supported tile explanations for one embedded bag."""

    full_output: Tensor
    leave_one_out: Tensor
    singleton: Tensor
    integrated_gradients: Tensor
    ig_baseline_output: Tensor
    ig_completeness_residual: Tensor
    ig_relative_completeness_error: Tensor
    attention: Tensor | None


def validate_pipeline(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor | None = None,
) -> None:
    """Validate that an aggregator/head pair can be explained by this module.

    Args:
        aggregator: The trained MIL aggregator.
        head: The trained raw-output prediction head.
        bag: Optional embedded bag with shape ``(num_tiles, feature_dim)``.

    Raises:
        TypeError: If the aggregator or head is not supported.
        ValueError: If dimensions, device, dtype, or bag contents are invalid.
    """
    if not isinstance(aggregator, (MeanPool, MaxPool, AttentionMIL)):
        raise TypeError(
            "Tile explainability supports only MeanPool, MaxPool, and "
            f"AttentionMIL; got {type(aggregator).__name__}."
        )
    if not isinstance(head, (LinearHead, MLPHead)):
        raise TypeError(
            "Tile explainability supports only LinearHead and MLPHead; "
            f"got {type(head).__name__}."
        )
    if aggregator.out_dim != head.in_dim:
        raise ValueError(
            "Aggregator/head dimension mismatch: "
            f"aggregator.out_dim={aggregator.out_dim}, head.in_dim={head.in_dim}."
        )
    if head.out_dim < 1:
        raise ValueError(f"The head must have at least one output; got {head.out_dim}.")

    if bag is None:
        return
    if bag.ndim != 2:
        raise ValueError(
            "An embedded bag must have shape (num_tiles, feature_dim); "
            f"got {tuple(bag.shape)}."
        )
    if bag.shape[0] < 1:
        raise ValueError("Cannot explain an empty bag.")
    if bag.shape[1] != aggregator.out_dim:
        raise ValueError(
            f"Bag feature dimension is {bag.shape[1]}, but the aggregator expects "
            f"{aggregator.out_dim}."
        )
    if not bag.is_floating_point():
        raise ValueError(f"Embedded bags must be floating point; got {bag.dtype}.")
    if not bool(torch.isfinite(bag).all()):
        raise ValueError("The embedded bag contains NaN or infinite values.")

    for component_name, component in (("aggregator", aggregator), ("head", head)):
        parameter = next(component.parameters(), None)
        if parameter is not None and parameter.device != bag.device:
            raise ValueError(
                f"Bag is on {bag.device}, but the {component_name} is on "
                f"{parameter.device}."
            )
        if parameter is not None and parameter.dtype != bag.dtype:
            raise ValueError(
                f"Bag has dtype {bag.dtype}, but the {component_name} has dtype "
                f"{parameter.dtype}."
            )


def forward_raw(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
) -> RawPipelineOutput:
    """Run an embedded bag through the aggregator and head in evaluation mode."""
    validate_pipeline(aggregator, head, bag)
    with _evaluation_mode(aggregator, head):
        return _forward_raw_unchecked(aggregator, head, bag)


def leave_one_out(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
    *,
    head_batch_size: int = 4096,
) -> LeaveOneOutResult:
    """Compute exact leave-one-tile-out necessity for every raw output.

    Pooling counterfactuals are constructed without rerunning the aggregator:
    mean pooling subtracts each tile from the sum; max pooling substitutes the
    feature-wise second maximum only when the removed tile is the unique maximum;
    gated attention subtracts the tile's normalized weighted contribution and
    renormalizes the remaining weights.  The resulting pooled vectors are passed
    through the complete linear or MLP head in chunks.

    A one-tile bag has no non-empty leave-one-out counterfactual.  Its
    ``without_tile`` and ``delta`` values are returned as NaN.
    """
    validate_pipeline(aggregator, head, bag)
    _validate_head_batch_size(head_batch_size)

    with _evaluation_mode(aggregator, head), torch.no_grad():
        full = _forward_raw_unchecked(aggregator, head, bag).scores
        if bag.shape[0] == 1:
            missing = torch.full(
                (1, head.out_dim),
                torch.nan,
                dtype=full.dtype,
                device=full.device,
            )
            return LeaveOneOutResult(full, missing, missing.clone())

        if isinstance(aggregator, MeanPool):
            without_tile = _mean_leave_one_out_outputs(head, bag, head_batch_size)
        elif isinstance(aggregator, MaxPool):
            without_tile = _max_leave_one_out_outputs(head, bag, head_batch_size)
        else:
            assert isinstance(aggregator, AttentionMIL)
            without_tile = _attention_leave_one_out_outputs(
                aggregator,
                head,
                bag,
                head_batch_size,
            )

        return LeaveOneOutResult(
            full_output=full,
            without_tile=without_tile,
            delta=full.unsqueeze(0) - without_tile,
        )


def singleton_sufficiency(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
    *,
    head_batch_size: int = 4096,
) -> Tensor:
    """Score every tile as a one-instance bag through the exact MIL pipeline."""
    validate_pipeline(aggregator, head, bag)
    _validate_head_batch_size(head_batch_size)

    with _evaluation_mode(aggregator, head), torch.no_grad():
        if isinstance(aggregator, AttentionMIL):
            # Softmax over a singleton is one, so this is exactly aggregator(tile[None]).
            singleton_pooled = aggregator.norm(bag)
        else:
            # Both mean and max of a singleton equal the tile itself.
            singleton_pooled = bag
        return _head_in_chunks(head, singleton_pooled, head_batch_size)


def integrated_gradients(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
    baseline: Tensor,
    *,
    steps: int = 32,
) -> IntegratedGradientsResult:
    """Compute manual Integrated Gradients through aggregator and head.

    ``baseline`` is one neutral embedding vector and is repeated to the bag size.
    Mean and attention pooling use ``steps + 1`` points and the trapezoidal rule.
    Max pooling instead has an exact pooled-space solution for the supported
    linear and two-layer ReLU heads.  This avoids evaluating the non-differentiable
    all-tiles-tied baseline and assigns each pooled-feature contribution to the
    tile or tiles that attain its maximum.

    Feature-level contributions are reduced immediately to signed tile scores,
    avoiding an ``(tiles, features, outputs)`` attribution allocation.
    """
    validate_pipeline(aggregator, head, bag)
    baseline = _validate_baseline(baseline, bag)
    if steps < 1:
        raise ValueError(f"Integrated Gradients steps must be positive; got {steps}.")

    with _evaluation_mode(aggregator, head):
        baseline_bag = baseline.unsqueeze(0).expand_as(bag)
        with torch.no_grad():
            full_output = _forward_raw_unchecked(aggregator, head, bag).scores
            baseline_output = _forward_raw_unchecked(
                aggregator, head, baseline_bag
            ).scores

        if isinstance(aggregator, MaxPool):
            with torch.no_grad():
                attributions = _max_integrated_gradients(head, bag, baseline)
        else:
            displacement = bag - baseline_bag
            attributions = torch.zeros(
                (bag.shape[0], head.out_dim),
                dtype=bag.dtype,
                device=bag.device,
            )
            for point in range(steps + 1):
                alpha = point / steps
                interpolated = (baseline_bag + alpha * displacement).detach()
                interpolated.requires_grad_(True)
                scores = _forward_raw_unchecked(aggregator, head, interpolated).scores
                trapezoid_weight = 0.5 if point in (0, steps) else 1.0

                for output_index in range(head.out_dim):
                    gradient = torch.autograd.grad(
                        scores[output_index],
                        interpolated,
                        retain_graph=output_index < head.out_dim - 1,
                    )[0]
                    tile_contribution = (gradient * displacement).sum(dim=1)
                    attributions[:, output_index].add_(
                        tile_contribution.detach(), alpha=trapezoid_weight / steps
                    )

    output_difference = full_output - baseline_output
    residual = output_difference - attributions.sum(dim=0)
    denominator = output_difference.abs().clamp_min(
        torch.finfo(output_difference.dtype).eps
    )
    relative_error = residual.abs() / denominator
    return IntegratedGradientsResult(
        attributions=attributions,
        full_output=full_output,
        baseline_output=baseline_output,
        completeness_residual=residual,
        relative_completeness_error=relative_error,
    )


def native_attention(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
) -> Tensor | None:
    """Return native gated-attention weights, or ``None`` for mean/max pooling."""
    validate_pipeline(aggregator, head, bag)
    if not isinstance(aggregator, AttentionMIL):
        return None
    with _evaluation_mode(aggregator), torch.no_grad():
        _, attention = aggregator(bag)
        return attention.detach()


def explain_bag(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
    baseline: Tensor,
    *,
    ig_steps: int = 32,
    head_batch_size: int = 4096,
) -> BagAttributionResult:
    """Run the complete supported attribution pass for one embedded slide."""
    loo = leave_one_out(
        aggregator,
        head,
        bag,
        head_batch_size=head_batch_size,
    )
    singleton = singleton_sufficiency(
        aggregator,
        head,
        bag,
        head_batch_size=head_batch_size,
    )
    ig = integrated_gradients(
        aggregator,
        head,
        bag,
        baseline,
        steps=ig_steps,
    )
    attention = native_attention(aggregator, head, bag)
    return BagAttributionResult(
        full_output=loo.full_output,
        leave_one_out=loo.delta,
        singleton=singleton,
        integrated_gradients=ig.attributions,
        ig_baseline_output=ig.baseline_output,
        ig_completeness_residual=ig.completeness_residual,
        ig_relative_completeness_error=ig.relative_completeness_error,
        attention=attention,
    )


def _forward_raw_unchecked(
    aggregator: Aggregator[Tensor],
    head: Head,
    bag: Tensor,
) -> RawPipelineOutput:
    pooled, attention = aggregator(bag)
    scores = head(pooled)
    if scores.ndim != 1 or scores.shape[0] != head.out_dim:
        raise ValueError(
            "The prediction head must return shape (out_dim,) for one bag; "
            f"got {tuple(scores.shape)}."
        )
    return RawPipelineOutput(scores=scores, pooled=pooled, attention=attention)


def _max_integrated_gradients(head: Head, bag: Tensor, baseline: Tensor) -> Tensor:
    """Compute exact max-pooling IG and map pooled features back to tiles.

    Repeating one baseline vector for every tile makes the max-pooled path
    ``baseline + alpha * (bag.amax(0) - baseline)`` for every ``alpha > 0``.
    The maximizing tile for each feature is therefore constant along the open
    path.  Computing IG in pooled space avoids the ambiguous gradient at
    ``alpha == 0``, where every baseline tile ties.
    """
    maximum = bag.amax(dim=0)
    feature_attributions = _head_feature_integrated_gradients(
        head,
        baseline,
        maximum,
    )

    maximum_count = torch.zeros(
        bag.shape[1],
        dtype=torch.int64,
        device=bag.device,
    )
    for tile_chunk in bag.split(_MAX_IG_TILE_BATCH_SIZE):
        maximum_count.add_((tile_chunk == maximum.unsqueeze(0)).sum(dim=0))
    maximum_count = maximum_count.clamp_min(1).to(dtype=bag.dtype)
    attribution_per_winner = feature_attributions / maximum_count.unsqueeze(1)
    attributions = torch.empty(
        (bag.shape[0], head.out_dim),
        dtype=bag.dtype,
        device=bag.device,
    )
    for start in range(0, bag.shape[0], _MAX_IG_TILE_BATCH_SIZE):
        stop = min(start + _MAX_IG_TILE_BATCH_SIZE, bag.shape[0])
        is_maximum = bag[start:stop] == maximum.unsqueeze(0)
        attributions[start:stop] = (
            is_maximum.to(dtype=bag.dtype) @ attribution_per_winner
        )
    return attributions


def _head_feature_integrated_gradients(
    head: Head,
    baseline: Tensor,
    features: Tensor,
) -> Tensor:
    """Return exact feature-by-output IG for a supported prediction head."""
    displacement = features - baseline
    if isinstance(head, LinearHead):
        return displacement.unsqueeze(1) * head.linear.weight.transpose(0, 1)

    assert isinstance(head, MLPHead)
    linear_layers = [layer for layer in head.net if isinstance(layer, nn.Linear)]
    if len(linear_layers) != 2:
        raise RuntimeError(
            "Exact max-pooling IG expects MLPHead to contain exactly two linear "
            f"layers; found {len(linear_layers)}."
        )
    first, second = linear_layers

    preactivation_start = first(baseline)
    preactivation_end = first(features)
    slope = preactivation_end - preactivation_start
    active_fraction = torch.empty_like(slope)

    increasing = slope > 0
    decreasing = slope < 0
    constant = ~(increasing | decreasing)
    active_fraction[increasing] = (
        preactivation_end[increasing] / slope[increasing]
    ).clamp(0, 1)
    active_fraction[decreasing] = (
        preactivation_start[decreasing] / -slope[decreasing]
    ).clamp(0, 1)
    active_fraction[constant] = (preactivation_start[constant] > 0).to(
        dtype=baseline.dtype
    )

    mean_gradient = second.weight @ (active_fraction.unsqueeze(1) * first.weight)
    return displacement.unsqueeze(1) * mean_gradient.transpose(0, 1)


def _mean_leave_one_out_outputs(
    head: Head,
    bag: Tensor,
    batch_size: int,
) -> Tensor:
    total = bag.sum(dim=0)
    predictions = []
    for start in range(0, bag.shape[0], batch_size):
        pooled = (total.unsqueeze(0) - bag[start : start + batch_size]) / (
            bag.shape[0] - 1
        )
        predictions.append(head(pooled))
    return torch.cat(predictions, dim=0)


def _max_leave_one_out_outputs(
    head: Head,
    bag: Tensor,
    batch_size: int,
) -> Tensor:
    maximum = bag.amax(dim=0)
    second_maximum = torch.topk(bag, k=2, dim=0).values[1]
    maximum_count = (bag == maximum.unsqueeze(0)).sum(dim=0)
    predictions = []
    for start in range(0, bag.shape[0], batch_size):
        is_unique_maximum = (
            bag[start : start + batch_size] == maximum.unsqueeze(0)
        ) & (maximum_count == 1).unsqueeze(0)
        pooled = torch.where(
            is_unique_maximum,
            second_maximum.unsqueeze(0),
            maximum.unsqueeze(0),
        )
        predictions.append(head(pooled))
    return torch.cat(predictions, dim=0)


def _attention_components(
    aggregator: AttentionMIL,
    bag: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    normalized = aggregator.norm(bag)
    gate = torch.tanh(aggregator.attention_v(normalized)) * torch.sigmoid(
        aggregator.attention_u(normalized)
    )
    logits = aggregator.attention_weights(gate).squeeze(-1)
    attention = torch.softmax(logits, dim=0)
    return normalized, logits, attention


def _attention_leave_one_out_outputs(
    aggregator: AttentionMIL,
    head: Head,
    bag: Tensor,
    batch_size: int,
) -> Tensor:
    normalized, _, attention = _attention_components(aggregator, bag)

    # Double precision greatly reduces cancellation when one tile dominates.
    normalized64 = normalized.double()
    attention64 = attention.double()
    pooled64 = (attention64.unsqueeze(1) * normalized64).sum(dim=0)
    denominator = 1.0 - attention64
    instability_threshold = max(
        torch.finfo(torch.float64).eps,
        32 * torch.finfo(bag.dtype).eps,
    )
    predictions = []
    for start in range(0, bag.shape[0], batch_size):
        stop = min(start + batch_size, bag.shape[0])
        chunk_attention = attention64[start:stop]
        chunk_normalized = normalized64[start:stop]
        chunk_denominator = denominator[start:stop]
        pooled_without64 = (
            pooled64.unsqueeze(0) - chunk_attention.unsqueeze(1) * chunk_normalized
        ) / chunk_denominator.unsqueeze(1)
        pooled_without = pooled_without64.to(dtype=bag.dtype)

        unstable = (~torch.isfinite(pooled_without).all(dim=1)) | (
            chunk_denominator <= instability_threshold
        )
        if bool(unstable.any()):
            for local_index in unstable.nonzero(as_tuple=False).flatten().tolist():
                tile_index = start + local_index
                keep = torch.arange(bag.shape[0], device=bag.device) != tile_index
                pooled_without[local_index] = aggregator(bag[keep])[0]
        predictions.append(head(pooled_without))

    return torch.cat(predictions, dim=0)


def _head_in_chunks(head: Head, features: Tensor, batch_size: int) -> Tensor:
    predictions = [
        head(features[start : start + batch_size])
        for start in range(0, features.shape[0], batch_size)
    ]
    result = torch.cat(predictions, dim=0)
    if result.ndim != 2 or result.shape[1] != head.out_dim:
        raise ValueError(
            "The prediction head must return shape (batch, out_dim) for pooled "
            f"batches; got {tuple(result.shape)}."
        )
    return result


def _validate_baseline(baseline: Tensor, bag: Tensor) -> Tensor:
    if baseline.ndim != 1 or baseline.shape[0] != bag.shape[1]:
        raise ValueError(
            "The baseline must be one embedding vector with shape "
            f"({bag.shape[1]},); got {tuple(baseline.shape)}."
        )
    if baseline.device != bag.device:
        raise ValueError(
            f"Baseline is on {baseline.device}, but the bag is on {bag.device}."
        )
    if baseline.dtype != bag.dtype:
        raise ValueError(
            f"Baseline has dtype {baseline.dtype}, but the bag has dtype {bag.dtype}."
        )
    if not bool(torch.isfinite(baseline).all()):
        raise ValueError("The baseline contains NaN or infinite values.")
    return baseline


def _validate_head_batch_size(head_batch_size: int) -> None:
    if head_batch_size < 1:
        raise ValueError(f"head_batch_size must be positive; got {head_batch_size}.")


@contextmanager
def _evaluation_mode(*modules: nn.Module) -> Iterator[None]:
    training_states = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        yield
    finally:
        for module, was_training in zip(modules, training_states, strict=True):
            module.train(was_training)


__all__ = [
    "BagAttributionResult",
    "IntegratedGradientsResult",
    "LeaveOneOutResult",
    "RawPipelineOutput",
    "SupportedAggregator",
    "SupportedHead",
    "explain_bag",
    "forward_raw",
    "integrated_gradients",
    "leave_one_out",
    "native_attention",
    "singleton_sufficiency",
    "validate_pipeline",
]
