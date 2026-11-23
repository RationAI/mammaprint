"""Evaluate and verify a single slide against its original frozen model."""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from ml.explainability.tile_removal import (
    MaskResult,
    class_removal_order,
    minimum_class_mask,
    pooled_index,
)
from ml.explainability.tile_removal_index import minimum_index_mask
from ml.explainability.tile_removal_inputs import sha256_file, tile_statistics
from ml.explainability.tile_removal_regions import (
    TileGrid,
    compact_class_mask,
    matched_controls,
    union_area,
)
from ml.explainability.tile_removal_viewer import (
    MaskCoverage,
    SlideGeometry,
    tile_rectangles,
    write_overlay,
)
from ml.models.encoders.tile_projector import TileProjector


if TYPE_CHECKING:
    from pathlib import Path

    from ml.explainability.tile_removal_inputs import PreparedModel, Slide


BUDGETS = {"1%": 1, "5%": 5, "10%": 10}
_BOUNDARY_TOLERANCE = 2e-6
LOG = logging.getLogger(__name__)


def prediction_class(index: float) -> str:
    return "A" if index >= 0 else "B"


def at_budget(number_of_tiles: int, percent: int) -> int:
    return min(number_of_tiles * percent // 100, number_of_tiles - 1)


def _same_score(first: float, second: float) -> bool:
    return math.isclose(first, second, abs_tol=_BOUNDARY_TOLERANCE, rel_tol=2e-5)


def _near_zero(value: float) -> bool:
    return abs(value) <= _BOUNDARY_TOLERANCE


@torch.inference_mode()
def evaluate_slide(
    prepared: PreparedModel,
    slide: Slide,
    *,
    options: dict[str, Any],
    root: Path,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    provenance = prepared.provenance
    row = {
        "model_id": provenance["model_id"],
        "slide_id": slide.slide_id,
        "patient_id": slide.patient_id,
        "recorded_class": slide.recorded_class,
        "recorded_index": slide.recorded_index,
        "recorded_index_class": (
            prediction_class(slide.recorded_index)
            if slide.recorded_index is not None
            else None
        ),
        "original_tile_count": None,
        "intact_index": None,
        "intact_class": None,
        "intact_abs_index": None,
        "target_class": None,
        "baseline_correct": None,
        "recorded_sign_disagrees": None,
        "saved_validation_index": prepared.predictions.get(slide.slide_id),
        "status": "ok",
        "geometry_status": "unavailable",
        "tumor_mask_status": "unavailable",
        "tissue_mask_status": "unavailable",
        "controls": {},
        "masks": [],
    }
    if row["recorded_class"] and row["recorded_index_class"]:
        row["recorded_sign_disagrees"] = (
            row["recorded_class"] != row["recorded_index_class"]
        )

    path = None
    try:
        path = prepared.bags.bag_path(slide.slide_id)
        if path is None:
            row["status"] = "missing_embedding"
            return row
        row["bag_sha256"] = sha256_file(path)
        # Keep the original row order and tolerate geometry-only schema issues.
        frame = pd.read_parquet(path)
        row["original_tile_count"] = len(frame)
        if frame.empty:
            row["status"] = "empty_bag"
            return row
        features = np.stack(frame["embedding"].to_numpy()).astype(np.float32)
        encoder = prepared.model.encoder
        dimension = (
            encoder.projection.in_features
            if isinstance(encoder, TileProjector)
            else encoder.out_dim
        )
        if features.shape != (len(frame), dimension) or not np.isfinite(features).all():
            raise ValueError("Invalid bag shape or values.")
        bag = torch.from_numpy(features).to(device)
    except Exception as error:
        LOG.warning(
            "%s / %s could not be read: %s",
            provenance["model_id"],
            slide.slide_id,
            error,
        )
        row["status"] = "read_error"
        return row
    finally:
        if (
            prepared.bags.local is None
            and path is not None
            and path.is_relative_to(prepared.bags.cache)
        ):
            path.unlink(missing_ok=True)

    model = prepared.model
    count = len(bag)
    ids = np.arange(count, dtype=np.int64)
    prediction, original_attention = model(bag)
    intact = float(prediction.item())
    scores, log_weights = tile_statistics(model, bag)
    if (
        not math.isfinite(intact)
        or not np.isfinite(scores).all()
        or not np.isfinite(log_weights).all()
    ):
        row["status"] = "nonfinite_model_output"
        return row
    math_intact = pooled_index(scores, log_weights)
    if not _same_score(math_intact, intact):
        row["status"] = "intact_parity_failed"
        return row

    intact_class = prediction_class(intact)
    row.update(
        intact_index=intact,
        intact_class=intact_class,
        intact_abs_index=abs(intact),
        baseline_correct=(
            intact_class == slide.recorded_class if slide.recorded_class else None
        ),
    )
    saved = row["saved_validation_index"]
    if saved is not None and not _same_score(intact, saved):
        row["status"] = "saved_prediction_parity_failed"
        return row
    if slide.recorded_class is None:
        row["status"] = "missing_class"
        if options["index_band"] is None or slide.recorded_index is None:
            return row
    target = (
        slide.recorded_class
        if not row["baseline_correct"]
        else ("B" if intact_class == "A" else "A")
    )
    row["target_class"] = target

    stride = provenance["stride"]
    extent = provenance["tile_extent"]
    wsi_loc = root / slide.wsi
    if wsi_root := options.get("wsi_root"):
        wsi_loc = root / wsi_root / slide.wsi.removeprefix("/")
    geometry = None
    grid = None
    try:
        geometry = SlideGeometry.from_wsi(wsi_loc, provenance["configured_mpp"])
        row["geometry_status"] = "verified_wsi"
        row["geometry"] = asdict(geometry)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
        LOG.info("WSI geometry unavailable for %s: %s", slide.slide_id, error)
        row["geometry_status"] = "unverified_wsi"
    try:
        rectangles = tile_rectangles(frame, extent, geometry)
        grid = TileGrid(rectangles, stride)
        level0 = geometry.level0_rectangles(rectangles) if geometry else None
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        LOG.info("Invalid tile geometry for %s: %s", slide.slide_id, error)
        grid = None
        row["geometry_status"] = "invalid_tile_coordinates"

    coverage: dict[str, MaskCoverage] = {}
    if geometry is not None and grid is not None and level0 is not None:
        for name in ("tumor", "tissue"):
            if directory := options.get(f"{name}_mask_dir"):
                try:
                    path = root / directory / f"{slide.slide_id}.tiff"
                    coverage[name] = MaskCoverage(
                        path, level0, geometry, options[f"{name}_mask_threshold"]
                    )
                    row[f"{name}_mask_status"] = "used"
                    row[f"{name}_mask_sha256"] = sha256_file(path)
                except (OSError, ValueError, IndexError) as error:
                    LOG.info(
                        "%s mask unavailable for %s: %s", name, slide.slide_id, error
                    )
    tumor_total = coverage["tumor"].totals(ids)[0] if "tumor" in coverage else None

    def coverage_values(indices: tuple[int, ...]) -> tuple[float, ...]:
        return tuple(layer.fraction(indices) for layer in coverage.values())

    def forward(keep: np.ndarray) -> float:
        pred, _ = model(bag[torch.from_numpy(keep).to(device)])
        return float(pred.item())

    def mask_forward(removed: tuple[int, ...]) -> tuple[float, np.ndarray]:
        keep = np.ones(count, dtype=bool)
        keep[list(removed)] = False
        return (forward(keep) if removed else intact), keep

    def native_class_candidate(
        result: MaskResult,
        target_class: str,
        *,
        limit: int,
        protected: tuple[int, ...] = (),
    ) -> MaskResult:
        """Retain a later verified flip when roundoff invalidates the first prefix."""
        candidate, _ = mask_forward(result.removed_ids)
        if prediction_class(candidate) == target_class or limit <= result.count:
            return result
        boundary = _near_zero(result.index) or _near_zero(candidate)
        if result.status == "infeasible" and not protected and limit == count - 1:
            boundary |= _near_zero(
                float(np.max(scores) if target_class == "A" else np.min(scores))
            )
        if not boundary:
            return result
        order = class_removal_order(
            scores, log_weights, ids, target_class, protected_ids=protected
        )
        upper = min(limit, len(order))
        # Try nearby counts first and then exponentially larger masks. Native
        # roundoff can perturb the ranking, so a rescued result is uncertified.
        offsets = [
            *range(1, 9),
            *(2**exponent for exponent in range(4, upper.bit_length() + 1)),
        ]
        counts = sorted({min(result.count + offset, upper) for offset in offsets})
        for size in counts:
            if size <= result.count:
                continue
            selected = order[:size]
            masked, keep = mask_forward(selected)
            expected = pooled_index(scores, log_weights, keep)
            if prediction_class(masked) == target_class and _same_score(
                masked, expected
            ):
                return MaskResult(selected, expected, "unresolved")
        return MaskResult(result.removed_ids, result.index, "unresolved")

    def append_mask(
        mask_id: str,
        kind: str,
        removed: tuple[int, ...],
        search_status: str,
        expected: float,
        *,
        predecessor: tuple[int, ...] | None = None,
        band: tuple[float, float] | None = None,
        window: tuple[int, int, int] | None = None,
        distance_bounds: tuple[float, float] | None = None,
        family: str | None = None,
        control_seed: int | None = None,
        render: bool = True,
    ) -> dict[str, Any]:
        masked, keep = mask_forward(removed)
        finite = math.isfinite(masked)
        masked_class = prediction_class(masked) if finite else None
        reached = finite and (
            band[0] <= masked <= band[1] if band else masked_class == target
        )
        verified = finite and _same_score(masked, expected)
        if not verified:
            verification_status = "forward_parity_failed"
        elif not reached:
            threshold_near = (
                min(abs(masked - bound) for bound in band) <= 2e-6
                if band
                else _near_zero(masked)
            )
            verification_status = (
                "boundary_unresolved"
                if threshold_near or search_status == "unresolved"
                else "target_not_reached"
            )
        elif search_status in {"minimum", "certified_minimum"} and not removed:
            verification_status = "verified_minimum"
        elif search_status in {"minimum", "certified_minimum"}:
            boundary_near = (
                min(abs(masked - bound) for bound in band) <= 2e-6
                if band
                else _near_zero(masked)
            )
            if predecessor is not None:
                prior_keep = np.ones(count, dtype=bool)
                prior_keep[list(predecessor)] = False
                prior = forward(prior_keep)
                prior_math = pooled_index(scores, log_weights, prior_keep)
                boundary_near |= (
                    not _same_score(prior, prior_math)
                    or prediction_class(prior) == target
                    or _near_zero(prior)
                )
            verification_status = (
                "minimum_unresolved" if boundary_near else "verified_minimum"
            )
            if kind == "compact" and verification_status == "verified_minimum":
                verification_status = "verified_window_minimum"
        elif search_status == "unresolved":
            verification_status = "minimum_unresolved"
        else:
            verification_status = "verified_target"
        control_values = coverage_values(removed)
        result = {
            "mask_id": mask_id,
            "kind": kind,
            "family": family,
            "control_seed": control_seed,
            "search_status": search_status,
            "verification_status": verification_status,
            "masked_index": masked if finite else None,
            "masked_class": masked_class,
            "deleted_count": len(removed),
            "deleted_fraction": len(removed) / count,
            "target_reached": bool(reached) if verified else None,
            "removed_tile_ids": sorted(removed),
            "tile_footprints_level0": None,
            "deleted_area_level0_px": None,
            "retained_tumor_fraction": None,
            "retained_attention_mass": None,
            "overlay_path": None,
            "viewer_url": None,
            "viewer_status": "unavailable",
            "overlay_expected": bool(render and removed),
            "window": list(window) if window is not None else None,
            "window_occupancy": len(removed) / window[2] ** 2
            if window is not None
            else None,
            "window_clipped": (
                window[0] + window[2] > grid.shape[1]
                or window[1] + window[2] > grid.shape[0]
                if window is not None and grid is not None
                else None
            ),
            "index_band": list(band) if band is not None else None,
            "distance_to_band": (
                max(0.0, band[0] - masked, masked - band[1])
                if band and finite
                else None
            ),
            "distance_bounds": list(distance_bounds)
            if distance_bounds is not None
            else None,
            "within_budgets": {
                key: len(removed) <= at_budget(count, percent) and bool(reached)
                for key, percent in BUDGETS.items()
            }
            if verified
            else None,
        }
        for name, value in zip(coverage, control_values, strict=True):
            result[f"deleted_{name}_coverage"] = value
        if original_attention is not None:
            result["retained_attention_mass"] = float(
                original_attention[torch.from_numpy(keep).to(device)].sum().item()
            )
        if tumor_total:
            result["retained_tumor_fraction"] = (
                coverage["tumor"].totals(ids[keep])[0] / tumor_total
            )
        if geometry is not None and grid is not None and level0 is not None:
            selected = level0[list(removed)]
            result["tile_footprints_level0"] = selected.tolist()
            result["deleted_area_level0_px"] = union_area(selected)
            if render and removed:
                # The overlay file and the blinded URL contain no mask-specific labels.
                opaque = hmac.new(
                    options["_review_key"],
                    f"overlay|{provenance['checkpoint_sha256']}|{slide.slide_id}|{mask_id}".encode(),
                    hashlib.sha256,
                ).hexdigest()[:32]
                relative = f"overlays/{opaque}.ome.tif"
                try:
                    write_overlay(
                        output / relative, rectangles[list(removed)], geometry
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    LOG.warning(
                        "Overlay failed for %s / %s: %s", slide.slide_id, mask_id, error
                    )
                    (output / relative).unlink(missing_ok=True)
                    result["viewer_status"] = "overlay_failed"
                else:
                    result["overlay_path"] = relative
                    result["viewer_status"] = "local_only"
        row["masks"].append(result)
        return result

    cap = at_budget(count, 10)
    if target is not None:
        exact = native_class_candidate(
            minimum_class_mask(scores, log_weights, ids, target),
            target,
            limit=count - 1,
        )
        exact_record = append_mask(
            "exact",
            "exact",
            exact.removed_ids,
            exact.status,
            exact.index,
            predecessor=exact.removed_ids[:-1] if exact.removed_ids else None,
        )
        families = []
        if exact_record["target_reached"] and exact.removed_ids:
            families.append(("exact", exact))
            alternative = minimum_class_mask(
                scores,
                log_weights,
                ids,
                target,
                max_removed=cap,
                protected_ids=exact.removed_ids,
            )
            alternative = native_class_candidate(
                alternative, target, limit=cap, protected=exact.removed_ids
            )
            if alternative.removed_ids:
                append_mask(
                    "alternative",
                    "alternative",
                    alternative.removed_ids,
                    alternative.status,
                    alternative.index,
                    predecessor=alternative.removed_ids[:-1],
                )
        if grid is not None:
            require_removal = prediction_class(math_intact) == target
            compact = compact_class_mask(
                scores,
                log_weights,
                ids,
                target,
                grid,
                max_removed=cap,
                require_removal=require_removal,
            )
            relaxed_window = compact_class_mask(
                scores,
                log_weights,
                ids,
                target,
                grid,
                max_removed=cap,
                require_removal=True,
                threshold=-_BOUNDARY_TOLERANCE
                if target == "A"
                else _BOUNDARY_TOLERANCE,
            )
            window_status = compact.status
            window_possible = (
                relaxed_window.status in {"minimum", "feasible", "unresolved"}
                and relaxed_window.count
            )
            if window_possible and (
                compact.status != "minimum" or relaxed_window.count < compact.count
            ):
                window_status = "unresolved"
                native, _ = mask_forward(relaxed_window.removed_ids)
                if prediction_class(native) == target and _same_score(
                    native, relaxed_window.index
                ):
                    compact = relaxed_window
            if require_removal and window_status == "minimum":
                window_status = "unresolved"
            append_mask(
                "compact",
                "compact",
                compact.removed_ids,
                window_status,
                compact.index,
                window=compact.window,
            )
            if compact.removed_ids:
                families.append(("compact", compact))
            for family, candidate in families:
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"{options['seed']}|{provenance['model_id']}|{slide.slide_id}|{family}".encode()
                    ).digest()[:4]
                )
                controls = matched_controls(
                    ids,
                    grid,
                    candidate.removed_ids,
                    compact=family == "compact",
                    seed=seed,
                    repeats=options["control_repeats"],
                    area_tolerance=options["area_tolerance"],
                    coverage=coverage_values if coverage else None,
                    coverage_tolerance=options["coverage_tolerance"],
                    window_size=candidate.window[2] if candidate.window else None,
                    area_rectangles=level0,
                )
                row["controls"][family] = {
                    "matched": len(controls),
                    "requested": options["control_repeats"],
                    "seed": seed,
                }
                for index, ids_control in enumerate(controls, start=1):
                    keep = ~np.isin(ids, ids_control)
                    append_mask(
                        f"{family}-control-{index:02}",
                        "control",
                        ids_control,
                        "matched_control",
                        pooled_index(scores, log_weights, keep),
                        family=family,
                        control_seed=seed,
                        render=(index == 1),
                    )
    if options["index_band"] is not None and slide.recorded_index is not None:
        bounds = (
            slide.recorded_index - options["index_band"],
            slide.recorded_index + options["index_band"],
        )
        band_result = minimum_index_mask(
            scores,
            log_weights,
            ids,
            *bounds,
            max_removed=cap,
            time_limit=options["index_time_limit"],
        )
        append_mask(
            "index",
            "index",
            band_result.removed_ids,
            band_result.status,
            band_result.index,
            band=bounds,
            distance_bounds=band_result.distance_bounds,
            render=False,
        )
    return row
