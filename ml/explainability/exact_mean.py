"""Exact, reference-relative evidence for an affine mean-pooling model."""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ml.explainability.attribution import validate_pipeline
from ml.models.aggregators.mean import MeanPool
from ml.models.heads.linear import LinearHead


@torch.no_grad()
def decompose_mean(
    aggregator: MeanPool,
    head: LinearHead,
    bag: Tensor,
    baseline: Tensor,
    *,
    batch_size: int = 1024,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return contributions (N,O), reference (O), and singleton scores (N,O).

    Chunked float64 arithmetic limits cancellation and bounds temporary memory.
    The independently evaluated full model remains the source of its prediction.
    Evaluation mode is required; no parameter or input gradients are computed.
    """
    if not isinstance(aggregator, MeanPool) or not isinstance(head, LinearHead):
        raise TypeError("Exact mean evidence requires MeanPool and LinearHead.")
    validate_pipeline(aggregator, head, bag)
    if head.training:
        raise ValueError("Exact mean evidence requires head.eval() (dropout disabled).")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if baseline.shape != (bag.shape[1],) or not torch.isfinite(baseline).all():
        raise ValueError("Reference must be one finite embedding vector.")
    weight = head.linear.weight.detach().to(device=bag.device, dtype=torch.float64)
    bias = head.linear.bias.detach().to(device=bag.device, dtype=torch.float64)
    reference_embedding = baseline.to(device=bag.device, dtype=torch.float64)
    reference = weight @ reference_embedding + bias
    scores = torch.cat(
        [chunk.to(torch.float64) @ weight.T + bias for chunk in bag.split(batch_size)]
    )
    contributions = (scores - reference) / bag.shape[0]
    if not torch.isfinite(contributions).all() or not torch.isfinite(reference).all():
        raise ValueError("Exact mean evidence is non-finite.")
    return contributions, reference, scores


def summarize_evidence(
    *,
    x: np.ndarray,
    y: np.ndarray,
    contributions: np.ndarray,
    singleton: np.ndarray,
    reference: float,
    prediction: float,
    target: str,
    region_size: int,
    tile_extent: int,
    top_tiles: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Partition every tile once by its origin; sums are in slide-output units."""
    if region_size < 1 or tile_extent < 1 or not 1 <= top_tiles <= 10:
        raise ValueError("Positive region/tile sizes and 1..10 top tiles are required.")
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError("Reconstruction tolerances must be finite and nonnegative.")
    c = np.asarray(contributions, dtype=np.float64)
    s = np.asarray(singleton, dtype=np.float64)
    x, y = np.asarray(x), np.asarray(y)
    if c.ndim != 1 or not len(c) or any(v.shape != c.shape for v in (s, x, y)):
        raise ValueError("Evidence needs equal nonempty one-dimensional tile arrays.")
    if not all(np.isfinite(v).all() for v in (c, s, x, y)) or not all(
        math.isfinite(v) for v in (reference, prediction)
    ):
        raise ValueError("Evidence must be finite.")
    positive = np.maximum(c, 0)
    negative = np.minimum(c, 0)
    up, down = float(positive.sum()), float(negative.sum())
    reconstructed = reference + up + down
    residual = prediction - reconstructed
    frame = pd.DataFrame(
        {
            "grid_x": x // region_size,
            "grid_y": y // region_size,
            "contribution": c,
            "up": positive,
            "down": negative,
            "x": x,
            "y": y,
        }
    )
    regions = []
    for (gx, gy), group in frame.groupby(["grid_x", "grid_y"], sort=True):
        regions.append(
            {
                "grid_x": int(gx),
                "grid_y": int(gy),
                "tile_count": len(group),
                "bounds": [
                    int(group.x.min()),
                    int(group.y.min()),
                    int(group.x.max()) + tile_extent,
                    int(group.y.max()) + tile_extent,
                ],
                "contribution": float(group.contribution.sum()),
                "up": float(group.up.sum()),
                "down": float(group.down.sum()),
            }
        )
    regions.sort(key=lambda region: -abs(region["contribution"]))
    selected = {}
    for direction, order in (
        ("up", np.argsort(-c, kind="stable")),
        ("down", np.argsort(c, kind="stable")),
    ):
        candidates = [
            int(i) for i in order if (c[i] > 0 if direction == "up" else c[i] < 0)
        ]
        selected[direction] = [
            {
                "tile_index": i,
                "x": int(x[i]),
                "y": int(y[i]),
                "contribution": float(c[i]),
                "singleton": float(s[i]),
            }
            for i in candidates[:top_tiles]
        ]
    counts, edges = np.histogram(s, bins=20)
    evidence = {
        "target": target,
        "units": "logit" if target == "luminal_a_logit" else "index",
        "tile_count": len(c),
        "reference": reference,
        "up": up,
        "down": down,
        "prediction": prediction,
        "reconstructed": reconstructed,
        "residual": residual,
        "absolute_error": abs(residual),
        "tolerance": atol + rtol * abs(prediction),
        "warning": abs(residual) > atol + rtol * abs(prediction),
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
        "regions": regions,
        "top_tiles": selected,
    }
    if target == "luminal_a_logit":
        evidence["probability"] = float(
            torch.sigmoid(torch.tensor(prediction, dtype=torch.float64))
        )
    return evidence


def attach_previews(
    evidence: dict[str, Any],
    *,
    slide_path: str,
    level: int,
    tile_extent: int,
) -> None:
    """Embed each selected crop once; image failures never invalidate evidence."""
    previews = evidence["previews"] = {}
    selected = {
        tile["tile_index"]: tile
        for target in evidence["targets"].values()
        for side in target["top_tiles"].values()
        for tile in side
    }
    try:
        with _open_slide(slide_path) as slide:
            for index, tile in selected.items():
                try:
                    image = slide.read_region_relative(
                        (tile["x"], tile["y"]), level, (tile_extent, tile_extent)
                    ).convert("RGB")
                    image.thumbnail((160, 160))
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=75)
                    previews[str(index)] = {
                        "status": "ok",
                        "jpeg": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    }
                except Exception as error:
                    previews[str(index)] = {
                        "status": "unavailable",
                        "error": str(error),
                    }
    except Exception as error:
        for index in selected:
            previews[str(index)] = {"status": "unavailable", "error": str(error)}


def _open_slide(path: str):
    from ratiopath.openslide import OpenSlide

    return OpenSlide(path)
