"""Cohort runner for post-hoc tile explanations."""

from __future__ import annotations

import logging
import math
import time
import traceback
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from ml.explainability.attribution import (
    forward_raw,
    integrated_gradients,
    leave_one_out,
    native_attention,
    singleton_sufficiency,
    validate_pipeline,
)
from ml.explainability.data import (
    EmbeddingSlide,
    SlideGeometry,
    instantiate_embedding_dataset,
    label_values,
    read_embedding_slide,
    read_slide_geometry,
    selected_level_spec,
    streaming_mean_embedding,
    target_names,
)
from ml.explainability.faithfulness import patch_flipping
from ml.explainability.outputs import (
    TILE_TABLE_COLUMNS,
    cohort_percentile_scale,
    make_tile_table,
    rasterize_tile_scores,
    write_manifest,
    write_pathologist_report,
    write_signed_ome_tiffs,
    write_summary_png,
    write_tile_geojson,
    write_tile_table,
)
from ml.explainability.viewer import ViewerLayer, build_xopat_review_url
from ml.models.encoders.identity import IdentityEncoder


if TYPE_CHECKING:
    from pathlib import Path

    from ml.models.module import MammaprintModule


log = logging.getLogger(__name__)


@dataclass
class RenderSlide:
    """Small retained state needed for cohort-calibrated output rendering."""

    slide_id: str
    x: np.ndarray
    y: np.ndarray
    slide_path: Path
    level: int
    scores: dict[str, dict[str, np.ndarray]]
    predictions: dict[str, float]
    labels: dict[str, float]


@dataclass
class CohortRunResult:
    """Paths and counts produced by :func:`run_cohort`."""

    output_dir: Path
    slides_path: Path
    slide_keys_path: Path
    pathologist_review_path: Path
    report_path: Path
    patch_flipping_path: Path | None
    manifest_path: Path
    successful_slides: int
    failed_slides: int
    metrics: dict[str, float]


@dataclass
class _CohortState:
    slide_rows: list[dict[str, Any]] = field(default_factory=list)
    patch_rows: list[dict[str, Any]] = field(default_factory=list)
    render_slides: list[RenderSlide] = field(default_factory=list)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def validate_module(module: MammaprintModule, config: DictConfig | None = None) -> None:
    """Enforce the v1 stored-embedding and model-family boundary."""
    if not isinstance(module.encoder, IdentityEncoder):
        raise TypeError(
            "The v1 explainer accepts stored embeddings only and requires "
            f"IdentityEncoder; got {type(module.encoder).__name__}."
        )
    validate_pipeline(module.aggregator, module.head)
    if config is None:
        return
    feature_dim = int(config.feature_dim)
    if module.encoder.out_dim != feature_dim:
        raise ValueError(
            "Recovered feature width does not match the identity encoder: "
            f"config.feature_dim={feature_dim}, encoder.out_dim={module.encoder.out_dim}."
        )
    target_names(str(config.label_mode), module.head.out_dim)


def run_cohort(
    *,
    config: DictConfig,
    module: MammaprintModule,
    output_dir: Path,
    provenance: dict[str, Any],
    slide_ids: list[str] | None = None,
    device: str = "auto",
) -> CohortRunResult:
    """Explain a configured split and materialize all raw/viewer artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tiles").mkdir(exist_ok=True)

    validate_module(module, config)
    torch_device = resolve_device(device)
    module.eval().to(torch_device)
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    explain_config = config.explain
    split = str(explain_config.split)
    dataset = instantiate_embedding_dataset(config, split)
    baseline_split = str(explain_config.ig.baseline_split)
    baseline_dataset = (
        dataset
        if split == baseline_split
        else instantiate_embedding_dataset(config, baseline_split)
    )
    baseline_cpu, baseline_tile_count, baseline_failures = streaming_mean_embedding(
        baseline_dataset,
        expected_feature_dim=int(config.feature_dim),
    )
    baseline = baseline_cpu.to(torch_device)

    level, level_spec = selected_level_spec(config)
    if dataset.level != level:
        raise ValueError(
            f"Dataset resolved level {dataset.level}, but data card resolved {level}."
        )
    feature_dim = int(config.feature_dim)
    if baseline.numel() != feature_dim or module.encoder.out_dim != feature_dim:
        raise ValueError(
            "Feature width mismatch among training baseline, recovered config, and "
            f"identity encoder: {baseline.numel()}, {feature_dim}, "
            f"{module.encoder.out_dim}."
        )
    targets = target_names(str(config.label_mode), module.head.out_dim)

    requested = set(slide_ids or [])
    available = set(dataset.slides["name"].astype(str))
    missing = requested - available
    if missing:
        raise KeyError(
            "Requested slide IDs are absent from the configured split: "
            + ", ".join(sorted(missing))
        )
    _validate_record_numbers(dataset.slides, requested)
    label_mode = str(config.label_mode)

    state = _CohortState()
    for index, row in dataset.slides.iterrows():
        slide_id = str(row["name"])
        if requested and slide_id not in requested:
            continue
        source_fields = _source_label_fields(row, label_mode)
        started = time.monotonic()
        try:
            slide = read_embedding_slide(dataset, int(index))
            if slide.embeddings.shape[1] != feature_dim:
                raise ValueError(
                    f"Slide {slide_id} has embedding width "
                    f"{slide.embeddings.shape[1]}, expected {feature_dim}."
                )
            render_slide, slide_row, patch_rows = _explain_slide(
                slide=slide,
                module=module,
                baseline=baseline,
                targets=targets,
                config=explain_config,
                output_dir=output_dir,
                device=torch_device,
            )
            attribution_runtime = time.monotonic() - started
            slide_row["attribution_runtime_seconds"] = attribution_runtime
            slide_row["runtime_seconds"] = attribution_runtime
            slide_row.update(source_fields)
            state.render_slides.append(render_slide)
            state.slide_rows.append(slide_row)
            state.patch_rows.extend(patch_rows)
        except Exception as error:
            log.exception("Explanation failed for slide %s", slide_id)
            state.slide_rows.append(
                {
                    **source_fields,
                    "slide_id": slide_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "runtime_seconds": time.monotonic() - started,
                    "overlay_status": "not_attempted",
                    "summary_status": "not_attempted",
                    "viewer_status": "not_attempted",
                }
            )
        finally:
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

    if not state.slide_rows:
        raise ValueError(f"No slides selected from split {split!r}.")

    scales = _cohort_scales(
        state.render_slides,
        percentile=float(explain_config.output.scale_percentile),
    )
    _render_outputs(
        state=state,
        output_dir=output_dir,
        level_spec=level_spec,
        scales=scales,
        config=explain_config,
        provenance=provenance,
    )

    slides_frame = pd.DataFrame(state.slide_rows)
    slides_path = output_dir / "slides.parquet"
    slides_frame.to_parquet(slides_path, index=False)
    slide_keys_path = output_dir / "slide_keys.csv"
    slide_keys_frame = _build_slide_key_table(state.slide_rows)
    slide_keys_frame.to_csv(
        slide_keys_path,
        index=False,
    )
    pathologist_review_path = output_dir / "pathologist_review.csv"
    _build_pathologist_review_table(slide_keys_frame).to_csv(
        pathologist_review_path,
        index=False,
    )
    patch_path: Path | None = None
    if state.patch_rows:
        patch_path = output_dir / "patch_flipping.parquet"
        pd.DataFrame(state.patch_rows).to_parquet(patch_path, index=False)

    successful = sum(row["status"] == "ok" for row in state.slide_rows)
    failed = len(state.slide_rows) - successful
    cohort_metrics = _cohort_metrics(slides_frame, slide_keys_frame)
    report_path = write_pathologist_report(
        output_dir / "report.html",
        slides=state.slide_rows,
        summary_paths={
            row["slide_id"]: output_dir / "summaries" / f"{row['slide_id']}.png"
            for row in state.slide_rows
            if (output_dir / "summaries" / f"{row['slide_id']}.png").is_file()
        },
        viewer_urls={
            row["slide_id"]: str(row["viewer_url"])
            for row in state.slide_rows
            if row.get("viewer_url")
        },
        title="MammaPrint tile-explainability review",
        metadata={
            "Split": split,
            "Aggregator": type(module.aggregator).__name__,
            "Head": type(module.head).__name__,
            "Source training run": provenance.get("source_run_id"),
            "Explanation run": provenance.get("explanation_run_id"),
        },
        warnings=_report_warnings(state.slide_rows, cohort_metrics),
    )
    enabled_methods = _enabled_methods(explain_config, module)
    manifest = {
        **provenance,
        "split": split,
        "slide_filter": sorted(requested),
        "supported_aggregators": ["mean", "max", "attention"],
        "supported_heads": ["linear", "mlp"],
        "targets": targets,
        "methods": enabled_methods,
        "primary_methods": [
            method
            for method in ("integrated_gradients", "leave_one_out")
            if method in enabled_methods
        ],
        "complementary_methods": [
            method for method in ("single",) if method in enabled_methods
        ],
        "diagnostic_methods": (
            ["attention"] if "attention" in enabled_methods else []
        ),
        "tile_table": {
            "layout": "one row per tile and target",
            "columns": list(TILE_TABLE_COLUMNS),
            "nullable_methods": [
                "integrated_gradients",
                "leave_one_out",
                "single",
                "attention",
            ],
        },
        "slide_key_file": {
            "path": slide_keys_path.name,
            "key": "record_num",
            "columns": list(SLIDE_KEY_COLUMNS),
            "task_values": ["class", "index", "both"],
            "contains_ground_truth_and_predictions": True,
        },
        "pathologist_review_file": {
            "path": pathologist_review_path.name,
            "key": "record_num",
            "columns": list(PATHOLOGIST_REVIEW_COLUMNS),
            "allowed_review_values": ["yes", "no", "uncertain"],
            "instructions": (
                "One row per slide. The reviewer fills prediction_makes_sense, "
                "highlighted_regions_make_sense, missed_relevant_regions, reviewer, "
                "and review_comment; generated prediction fields must remain unchanged."
            ),
        },
        "pathologist_preview": {
            "path": report_path.name,
            "mlflow_run_url": provenance.get("mlflow_run_url"),
            "xopat_ui_uri": str(
                provenance.get("xopat_ui_uri") or explain_config.output.xopat_ui_uri
            ),
            "interactive_layers": (
                "MLflow-hosted OME-TIFF masks are linked into xOpat using the "
                "RationAI report mount convention. A slide without a viewer-accessible "
                "WSI remains available through its static summary and raw tables."
            ),
        },
        "baseline": {
            "kind": "training_split_tile_mean",
            "split": baseline_split,
            "tile_count": baseline_tile_count,
            "skipped_slides": list(baseline_failures),
        },
        "level": level,
        "tile_extent": int(level_spec["tile_extent"]),
        "stride": int(level_spec["stride"]),
        "configured_mpp": float(level_spec["mpp"]),
        "level_geometry": {
            "coordinate_space": "selected_wsi_level_pixels",
            "overlay_dimensions_source": "original_wsi",
            "geojson_coordinate_space": "level_0_pixels",
            "per_slide_geometry_columns": [
                "wsi_level_width",
                "wsi_level_height",
                "wsi_level_downsample",
                "wsi_level_mpp_x",
                "wsi_level_mpp_y",
                "wsi_level_mpp_source_x",
                "wsi_level_mpp_source_y",
            ],
        },
        "visualization_scaling": {
            "kind": "cohort_absolute_percentile",
            "percentile": float(explain_config.output.scale_percentile),
            "per_slide_normalization": False,
            "output_dtype": "uint8",
        },
        "visualization_scales": {
            f"{target}/{method}": scale
            for (target, method), scale in sorted(scales.items())
        },
        "score_semantics": {
            "targets": {
                "luminal_a_logit": (
                    "positive output or contribution favors Luminal A; negative favors "
                    "Luminal B"
                ),
                "mammaprint_index": (
                    "positive contribution raises the predicted index; negative lowers it"
                ),
            },
            "methods": {
                "integrated_gradients": (
                    "signed contribution relative to the repeated training-mean "
                    "embedding baseline"
                ),
                "leave_one_out": (
                    "signed full-bag output minus output after removing the tile"
                ),
                "single": (
                    "absolute raw output from a one-tile bag; its sign is not a causal "
                    "change from the full bag"
                ),
                "attention": (
                    "non-negative, non-target-specific diagnostic weight; not a "
                    "primary explanation"
                ),
            },
        },
        "counts": {"successful_slides": successful, "failed_slides": failed},
        "cohort_metrics": cohort_metrics,
        "explain_config": OmegaConf.to_container(explain_config, resolve=True),
    }
    manifest_path = write_manifest(output_dir / "manifest.json", manifest)
    return CohortRunResult(
        output_dir=output_dir,
        slides_path=slides_path,
        slide_keys_path=slide_keys_path,
        pathologist_review_path=pathologist_review_path,
        report_path=report_path,
        patch_flipping_path=patch_path,
        manifest_path=manifest_path,
        successful_slides=successful,
        failed_slides=failed,
        metrics=cohort_metrics,
    )


def _explain_slide(
    *,
    slide: EmbeddingSlide,
    module: MammaprintModule,
    baseline: Tensor,
    targets: list[str],
    config: DictConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[RenderSlide, dict[str, Any], list[dict[str, Any]]]:
    bag = slide.embeddings.to(device)
    full_output = forward_raw(module.aggregator, module.head, bag).scores
    method_tensors: dict[str, Tensor] = {}
    ig_baseline: Tensor | None = None
    ig_residual: Tensor | None = None

    batch_size = int(config.counterfactual_batch_size)
    if bool(config.leave_one_out.enabled) and bag.shape[0] > 1:
        loo = leave_one_out(
            module.aggregator,
            module.head,
            bag,
            head_batch_size=batch_size,
        )
        method_tensors["leave_one_out"] = loo.delta
    elif bool(config.leave_one_out.enabled):
        # An empty counterfactual does not exist. Keep the raw table finite and
        # record the undefined method explicitly in the slide summary below.
        log.warning(
            "Slide %s has one tile; leave-one-out is undefined.", slide.slide_id
        )
    if bool(config.single.enabled):
        method_tensors["single"] = singleton_sufficiency(
            module.aggregator,
            module.head,
            bag,
            head_batch_size=batch_size,
        )
    if bool(config.ig.enabled):
        ig = integrated_gradients(
            module.aggregator,
            module.head,
            bag,
            baseline=baseline,
            steps=int(config.ig.steps),
        )
        method_tensors["integrated_gradients"] = ig.attributions
        ig_baseline = ig.baseline_output
        ig_residual = ig.completeness_residual
    if bool(config.attention.enabled):
        attention = native_attention(module.aggregator, module.head, bag)
        if attention is not None:
            method_tensors["attention"] = attention[:, None].expand(-1, len(targets))

    predictions = {
        target: float(full_output[target_index].detach().cpu())
        for target_index, target in enumerate(targets)
    }
    labels_list = label_values(slide.label)
    labels = {target: labels_list[index] for index, target in enumerate(targets)}
    scores = {
        target: {
            method: _as_numpy(values[:, target_index])
            for method, values in method_tensors.items()
        }
        for target_index, target in enumerate(targets)
    }

    tile_frame = make_tile_table(
        slide_id=slide.slide_id,
        x=slide.x,
        y=slide.y,
        attributions=scores,
        predictions=predictions,
        labels=labels,
    )
    tile_path = write_tile_table(
        tile_frame,
        output_dir / "tiles" / f"{slide.slide_id}.parquet",
    )

    slide_row: dict[str, Any] = {
        "slide_id": slide.slide_id,
        "status": "ok",
        "error_type": None,
        "error": None,
        "traceback": None,
        "tile_count": bag.shape[0],
        "level": slide.level,
        "slide_path": str(slide.slide_path),
        "tile_parquet": str(tile_path.relative_to(output_dir)),
        "overlay_status": "pending",
    }
    if bool(config.leave_one_out.enabled) and bag.shape[0] == 1:
        slide_row["leave_one_out_status"] = "undefined_single_tile_bag"
    tolerance = float(config.ig.completeness_tolerance)
    for target_index, target in enumerate(targets):
        prediction = predictions[target]
        slide_row[f"prediction_raw/{target}"] = prediction
        slide_row[f"label/{target}"] = labels[target]
        if target == "luminal_a_logit":
            slide_row[f"prediction_probability/{target}"] = _sigmoid(prediction)
        if ig_baseline is not None and ig_residual is not None:
            base = float(ig_baseline[target_index].detach().cpu())
            residual = float(ig_residual[target_index].detach().cpu())
            denominator = max(abs(prediction - base), 1e-8)
            relative = abs(residual) / denominator
            slide_row[f"ig_baseline_raw/{target}"] = base
            slide_row[f"ig_completeness_residual/{target}"] = residual
            slide_row[f"ig_completeness_relative/{target}"] = relative
            slide_row[f"ig_completeness_warning/{target}"] = relative > tolerance
            if relative > tolerance:
                log.warning(
                    "Slide %s target %s has IG relative completeness error %.2f%% "
                    "(tolerance %.2f%%).",
                    slide.slide_id,
                    target,
                    100 * relative,
                    100 * tolerance,
                )

    patch_rows: list[dict[str, Any]] = []
    if bool(config.patch_flipping.enabled):
        for method, tile_scores in method_tensors.items():
            # Recreate the same deterministic random comparator for every method on
            # one slide, so method-to-method faithfulness comparisons are paired.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_slide_seed(int(config.seed), slide.slide_id))
            result = patch_flipping(
                module.aggregator,
                module.head,
                bag,
                tile_scores,
                baseline,
                percentage_step=int(config.patch_flipping.step_percent),
                random_repeats=int(config.patch_flipping.random_repeats),
                head_batch_size=batch_size,
                generator=generator,
            )
            for target_index, target in enumerate(targets):
                patch_rows.extend(
                    _patch_records(
                        slide_id=slide.slide_id,
                        target=target,
                        target_index=target_index,
                        method=method,
                        result=result,
                    )
                )
                slide_row[f"faithfulness_srg/{target}/{method}"] = (
                    _target_scalar(result.srg, target_index)
                )
                slide_row[f"faithfulness_descending_auc/{target}/{method}"] = (
                    _target_scalar(result.descending_auc, target_index)
                )
                slide_row[f"faithfulness_ascending_auc/{target}/{method}"] = (
                    _target_scalar(result.ascending_auc, target_index)
                )
                random_auc = _as_numpy(result.random_auc)[:, target_index]
                slide_row[f"faithfulness_random_auc_mean/{target}/{method}"] = float(
                    np.mean(random_auc)
                )
                slide_row[f"faithfulness_random_auc_std/{target}/{method}"] = float(
                    np.std(random_auc)
                )

    return (
        RenderSlide(
            slide_id=slide.slide_id,
            x=slide.x,
            y=slide.y,
            slide_path=slide.slide_path,
            level=slide.level,
            scores=scores,
            predictions=predictions,
            labels=labels,
        ),
        slide_row,
        patch_rows,
    )


def _patch_records(
    *,
    slide_id: str,
    target: str,
    target_index: int,
    method: str,
    result: Any,
) -> list[dict[str, Any]]:
    fractions = _as_numpy(result.fractions)
    records: list[dict[str, Any]] = []
    curves = {
        "descending": _as_numpy(result.descending)[target_index],
        "ascending": _as_numpy(result.ascending)[target_index],
    }
    for order, curve in curves.items():
        for fraction, prediction in zip(fractions, curve, strict=True):
            records.append(
                {
                    "slide_id": slide_id,
                    "target": target,
                    "method": method,
                    "order": order,
                    "random_repeat": None,
                    "fraction_removed": float(fraction),
                    "prediction_raw": float(prediction),
                }
            )
    random_curves = _as_numpy(result.random)
    for repeat, curve_matrix in enumerate(random_curves):
        for fraction, prediction in zip(
            fractions, curve_matrix[target_index], strict=True
        ):
            records.append(
                {
                    "slide_id": slide_id,
                    "target": target,
                    "method": method,
                    "order": "random",
                    "random_repeat": repeat,
                    "fraction_removed": float(fraction),
                    "prediction_raw": float(prediction),
                }
            )
    return records


def _cohort_scales(
    slides: list[RenderSlide],
    *,
    percentile: float,
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for slide in slides:
        for target, methods in slide.scores.items():
            for method, scores in methods.items():
                grouped.setdefault((target, method), []).append(scores)
    return {
        key: cohort_percentile_scale(score_sets, percentile=percentile)
        for key, score_sets in grouped.items()
    }


def _render_outputs(
    *,
    state: _CohortState,
    output_dir: Path,
    level_spec: dict[str, Any],
    scales: dict[tuple[str, str], float],
    config: DictConfig,
    provenance: dict[str, Any],
) -> None:
    tile_extent = int(level_spec["tile_extent"])
    configured_mpp = float(level_spec["mpp"])
    row_by_slide = {row["slide_id"]: row for row in state.slide_rows}
    for slide in state.render_slides:
        row = row_by_slide[slide.slide_id]
        render_started = time.monotonic()
        try:
            geometry = read_slide_geometry(
                slide.slide_path,
                slide.level,
                configured_mpp,
            )
            if bool((slide.x < 0).any()) or bool((slide.y < 0).any()) or bool(
                (slide.x >= geometry.width).any()
            ) or bool((slide.y >= geometry.height).any()):
                raise ValueError(
                    "Tile coordinates fall outside the selected WSI level; refusing "
                    "to create a plausibly aligned mask from mismatched geometry."
                )
            for target, methods in slide.scores.items():
                for method, scores in methods.items():
                    scale = scales[(target, method)]
                    mask_dir = output_dir / "masks" / method / target
                    write_signed_ome_tiffs(
                        mask_dir,
                        slide.slide_id,
                        slide.x,
                        slide.y,
                        scores,
                        geometry=geometry,
                        tile_extent=tile_extent,
                        scale=scale,
                        tile_size=int(config.output.tiff_tile_size),
                    )
                    write_tile_geojson(
                        output_dir
                        / "geojson"
                        / method
                        / target
                        / f"{slide.slide_id}.geojson",
                        slide_id=slide.slide_id,
                        x=slide.x,
                        y=slide.y,
                        scores=scores,
                        geometry=geometry,
                        tile_extent=tile_extent,
                        target=target,
                        method=method,
                        fraction=float(config.output.geojson_fraction),
                    )
            row["overlay_status"] = "ok"
            row["overlay_error"] = None
            row["wsi_level_width"] = geometry.width
            row["wsi_level_height"] = geometry.height
            row["wsi_level_downsample"] = geometry.downsample
            row["wsi_level_mpp_x"] = geometry.mpp_x
            row["wsi_level_mpp_y"] = geometry.mpp_y
            row["wsi_level_mpp_source_x"] = geometry.mpp_source_x
            row["wsi_level_mpp_source_y"] = geometry.mpp_source_y
            _attach_viewer_url(
                row=row,
                slide=slide,
                provenance=provenance,
                artifact_root=str(config.output.artifact_path),
                xopat_ui_uri=str(
                    provenance.get("xopat_ui_uri") or config.output.xopat_ui_uri
                ),
            )
            try:
                _write_slide_summary(
                    state=state,
                    slide=slide,
                    geometry=geometry,
                    tile_extent=tile_extent,
                    output_dir=output_dir,
                    slide_row=row,
                    scales=scales,
                )
                row["summary_status"] = "ok"
                row["summary_error"] = None
            except Exception as summary_error:
                log.exception("Summary failed for slide %s", slide.slide_id)
                row["summary_status"] = "failed"
                row["summary_error"] = (
                    f"{type(summary_error).__name__}: {summary_error}"
                )
        except FileNotFoundError as error:
            log.warning("Viewer output unavailable for slide %s: %s", slide.slide_id, error)
            row["overlay_status"] = "unavailable"
            row["overlay_error"] = f"{type(error).__name__}: {error}"
            row["summary_status"] = "unavailable"
            row["summary_error"] = row["overlay_error"]
            row["viewer_status"] = "unavailable"
            row["viewer_error"] = row["overlay_error"]
        except Exception as error:
            log.exception("Viewer output failed for slide %s", slide.slide_id)
            row["overlay_status"] = "failed"
            row["overlay_error"] = f"{type(error).__name__}: {error}"
            row["summary_status"] = "not_attempted"
            row["summary_error"] = row["overlay_error"]
            row["viewer_status"] = "not_attempted"
            row["viewer_error"] = row["overlay_error"]
        finally:
            overlay_runtime = time.monotonic() - render_started
            row["overlay_runtime_seconds"] = overlay_runtime
            row["runtime_seconds"] = float(row["runtime_seconds"]) + overlay_runtime


def _attach_viewer_url(
    *,
    row: dict[str, Any],
    slide: RenderSlide,
    provenance: dict[str, Any],
    artifact_root: str,
    xopat_ui_uri: str,
) -> None:
    """Attach an xOpat deep-link without making viewer access inference-critical."""
    experiment_id = provenance.get("explanation_experiment_id")
    run_id = provenance.get("explanation_run_id")
    if not experiment_id or not run_id:
        row["viewer_status"] = "unavailable"
        row["viewer_error"] = (
            "Explanation run identity is unavailable (for example, a local cohort run)."
        )
        row["viewer_url"] = None
        return

    try:
        row["viewer_url"] = build_xopat_review_url(
            xopat_ui_uri=xopat_ui_uri,
            experiment_id=str(experiment_id),
            run_id=str(run_id),
            artifact_root=artifact_root,
            slide_id=str(row.get("record_num") or slide.slide_id),
            slide_path=str(slide.slide_path),
            layers=_viewer_layers(slide),
        )
    except (TypeError, ValueError) as error:
        log.warning("Interactive viewer unavailable for slide %s: %s", slide.slide_id, error)
        row["viewer_status"] = "unavailable"
        row["viewer_error"] = f"{type(error).__name__}: {error}"
        row["viewer_url"] = None
        return

    row["viewer_status"] = "ok"
    row["viewer_error"] = None


def _viewer_layers(slide: RenderSlide) -> list[ViewerLayer]:
    """Describe every generated TIFF using clinically honest layer names."""
    layers: list[ViewerLayer] = []
    first_target = next(iter(slide.scores), None)
    attention_added = False
    colors = {
        "integrated_gradients": ("#d73027", "#4575b4"),
        "leave_one_out": ("#ef476f", "#118ab2"),
        "single": ("#f28e2b", "#9467bd"),
    }
    for target, methods in slide.scores.items():
        target_label = _friendly_target_name(target)
        for method in ("integrated_gradients", "leave_one_out", "single"):
            if method not in methods:
                continue
            positive_color, negative_color = colors[method]
            method_label = {
                "integrated_gradients": "Integrated gradients",
                "leave_one_out": "Tile removal",
                "single": "Single-tile output",
            }[method]
            if method == "single":
                directions = ("positive", "negative")
            else:
                directions = ("raises", "lowers")
            mask_root = f"masks/{method}/{target}"
            initially_visible = method == "integrated_gradients" and target == first_target
            layers.extend(
                (
                    ViewerLayer(
                        name=f"{method_label} — {directions[0]} {target_label}",
                        artifact_path=(
                            f"{mask_root}/{slide.slide_id}_positive.ome.tif"
                        ),
                        color=positive_color,
                        visible=initially_visible,
                    ),
                    ViewerLayer(
                        name=f"{method_label} — {directions[1]} {target_label}",
                        artifact_path=(
                            f"{mask_root}/{slide.slide_id}_negative.ome.tif"
                        ),
                        color=negative_color,
                        visible=initially_visible,
                    ),
                )
            )
        if "attention" in methods and not attention_added:
            layers.append(
                ViewerLayer(
                    name="Attention weight (diagnostic, not target-specific)",
                    artifact_path=(
                        f"masks/attention/{target}/{slide.slide_id}_positive.ome.tif"
                    ),
                    color="#f2c14e",
                    visible=False,
                )
            )
            attention_added = True
    return layers


def _friendly_target_name(target: str) -> str:
    return {
        "luminal_a_logit": "Luminal A evidence",
        "mammaprint_index": "MammaPrint index",
    }.get(target, target.replace("_", " "))


def _write_slide_summary(
    *,
    state: _CohortState,
    slide: RenderSlide,
    geometry: SlideGeometry,
    tile_extent: int,
    output_dir: Path,
    slide_row: dict[str, Any],
    scales: dict[tuple[str, str], float],
) -> None:
    preview_downsample = max(1, math.ceil(max(geometry.width, geometry.height) / 1024))
    primary_methods = {"integrated_gradients", "leave_one_out"}
    selected = [
        (target, method, scores)
        for target, methods in slide.scores.items()
        for method, scores in methods.items()
        if method in primary_methods
    ]
    if not selected:
        selected = [
            (target, method, scores)
            for target, methods in slide.scores.items()
            for method, scores in methods.items()
        ]
    preview_rasters = {
        _attribution_panel_name(target, method): rasterize_tile_scores(
            slide.x,
            slide.y,
            scores,
            geometry=geometry,
            tile_extent=tile_extent,
            raster_downsample=preview_downsample,
        )
        for target, method, scores in selected
    }
    attribution_maps = {
        name: raster.values for name, raster in preview_rasters.items()
    }
    attribution_coverages = {
        name: raster.coverage for name, raster in preview_rasters.items()
    }
    attribution_limits = {
        _attribution_panel_name(target, method): scales[(target, method)]
        for target, method, _ in selected
    }
    thumbnail = _read_thumbnail(slide.slide_path)
    prediction_text = ", ".join(
        f"{target}={value:.4g}" for target, value in slide.predictions.items()
    )
    label_text = ", ".join(
        f"{target}={value:.4g}" for target, value in slide.labels.items()
    )
    faithfulness = {
        key.removeprefix("faithfulness_srg/"): float(value)
        for key, value in slide_row.items()
        if key.startswith("faithfulness_srg/") and value is not None
    }
    curves = _summary_curves(state.patch_rows, slide.slide_id)
    write_summary_png(
        output_dir / "summaries" / f"{slide.slide_id}.png",
        slide_id=slide.slide_id,
        record_num=slide_row.get("record_num"),
        thumbnail=thumbnail,
        attribution_maps=attribution_maps,
        attribution_coverages=attribution_coverages,
        attribution_limits=attribution_limits,
        prediction_text=prediction_text,
        label_text=label_text,
        faithfulness=faithfulness,
        curves=curves,
    )


def _read_thumbnail(slide_path: Path) -> np.ndarray:
    from openslide import OpenSlide

    with OpenSlide(str(slide_path)) as slide:
        thumbnail = slide.get_thumbnail((1024, 1024)).convert("RGB")
    return np.asarray(thumbnail)


def _summary_curves(
    patch_rows: list[dict[str, Any]],
    slide_id: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    relevant = [
        row
        for row in patch_rows
        if row["slide_id"] == slide_id and row["order"] in {"ascending", "descending"}
    ]
    if not relevant:
        return {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    keys = sorted(
        {
            (str(row["target"]), str(row["method"]), str(row["order"]))
            for row in relevant
        }
    )
    for target, method, order in keys:
        points = sorted(
            (
                float(row["fraction_removed"]),
                float(row["prediction_raw"]),
            )
            for row in relevant
            if row["target"] == target
            and row["method"] == method
            and row["order"] == order
        )
        curves[f"{target}/{method}/{order}"] = (
            np.asarray([point[0] for point in points]),
            np.asarray([point[1] for point in points]),
        )
    return curves


def _attribution_panel_name(target: str, method: str) -> str:
    if method == "attention":
        return "Attention weight (diagnostic; not target-specific)"
    method_label = {
        "integrated_gradients": "Integrated gradients",
        "leave_one_out": "Tile removal",
        "single": "Single-tile output (diagnostic)",
    }.get(method, method.replace("_", " ").title())
    return f"{_friendly_target_name(target)} · {method_label}"


SLIDE_KEY_COLUMNS = (
    "record_num",
    "slide_id",
    "task",
    "status",
    "class",
    "class_label",
    "mammaprint_index",
    "prediction_class",
    "prediction_class_label",
    "prediction_luminal_a_probability",
    "prediction_luminal_a_logit",
    "prediction_mammaprint_index",
)

PATHOLOGIST_REVIEW_COLUMNS = (
    *SLIDE_KEY_COLUMNS,
    "prediction_makes_sense",
    "highlighted_regions_make_sense",
    "missed_relevant_regions",
    "reviewer",
    "review_comment",
)


def _validate_record_numbers(slides: pd.DataFrame, requested: set[str]) -> None:
    if "record_num" not in slides.columns:
        raise KeyError(
            "The data mapping has no 'record_num' column required for slide_keys.csv."
        )
    selected = slides if not requested else slides[slides["name"].astype(str).isin(requested)]
    invalid = [
        str(row["name"])
        for _, row in selected.iterrows()
        if pd.isna(row["record_num"]) or not str(row["record_num"]).strip()
    ]
    if invalid:
        raise ValueError(
            "Selected slides have missing record_num values: "
            + ", ".join(sorted(invalid))
        )


def _source_label_fields(row: pd.Series, label_mode: str) -> dict[str, Any]:
    task = {"type": "class", "index": "index", "both": "both"}.get(label_mode)
    if task is None:
        raise ValueError(f"Unsupported label_mode {label_mode!r}.")
    include_class = label_mode in {"type", "both"}
    include_index = label_mode in {"index", "both"}
    return {
        "record_num": _optional_python_value(row.get("record_num")),
        "task": task,
        "class": _optional_python_value(row.get("type")) if include_class else None,
        "class_label": (
            _optional_python_value(row.get("type_label")) if include_class else None
        ),
        "mammaprint_index": (
            _optional_python_value(row.get("mammaprint_index"))
            if include_index
            else None
        ),
    }


def _build_slide_key_table(slide_rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in slide_rows:
        logit = row.get("prediction_raw/luminal_a_logit")
        probability = row.get("prediction_probability/luminal_a_logit")
        predicted_class_label = None if logit is None else int(float(logit) >= 0)
        records.append(
            {
                "record_num": row.get("record_num"),
                "slide_id": row["slide_id"],
                "task": row.get("task"),
                "status": row["status"],
                "class": row.get("class"),
                "class_label": row.get("class_label"),
                "mammaprint_index": row.get("mammaprint_index"),
                "prediction_class": (
                    None
                    if predicted_class_label is None
                    else "a luminal"
                    if predicted_class_label == 1
                    else "b luminal"
                ),
                "prediction_class_label": predicted_class_label,
                "prediction_luminal_a_probability": probability,
                "prediction_luminal_a_logit": logit,
                "prediction_mammaprint_index": row.get(
                    "prediction_raw/mammaprint_index"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SLIDE_KEY_COLUMNS)


def _build_pathologist_review_table(slide_keys: pd.DataFrame) -> pd.DataFrame:
    """Seed an editable review sheet without modifying generated predictions."""
    missing = set(SLIDE_KEY_COLUMNS) - set(slide_keys.columns)
    if missing:
        raise ValueError(
            "Slide-key table is missing columns required for pathologist review: "
            f"{sorted(missing)}."
        )
    review = slide_keys.loc[:, SLIDE_KEY_COLUMNS].copy()
    for column in PATHOLOGIST_REVIEW_COLUMNS[len(SLIDE_KEY_COLUMNS) :]:
        review[column] = ""
    return review.loc[:, PATHOLOGIST_REVIEW_COLUMNS]


def _cohort_metrics(
    slides: pd.DataFrame,
    slide_keys: pd.DataFrame,
) -> dict[str, float]:
    """Aggregate only cohort-level values suitable for MLflow metrics."""
    metrics: dict[str, float] = {}
    class_rows = slide_keys.dropna(
        subset=["class_label", "prediction_class_label"]
    )
    if not class_rows.empty:
        truth = pd.to_numeric(class_rows["class_label"], errors="coerce")
        predicted = pd.to_numeric(
            class_rows["prediction_class_label"], errors="coerce"
        )
        valid = truth.notna() & predicted.notna()
        if valid.any():
            metrics["classification_accuracy"] = float(
                (truth[valid] == predicted[valid]).mean()
            )

    index_rows = slide_keys.dropna(
        subset=["mammaprint_index", "prediction_mammaprint_index"]
    )
    if not index_rows.empty:
        truth_index = pd.to_numeric(
            index_rows["mammaprint_index"], errors="coerce"
        )
        predicted_index = pd.to_numeric(
            index_rows["prediction_mammaprint_index"], errors="coerce"
        )
        valid = truth_index.notna() & predicted_index.notna()
        if valid.any():
            metrics["regression_mae"] = float(
                (truth_index[valid] - predicted_index[valid]).abs().mean()
            )

    warning_columns = [
        column
        for column in slides.columns
        if str(column).startswith("ig_completeness_warning/")
    ]
    if warning_columns:
        warnings = slides.loc[:, warning_columns].fillna(False).astype(bool)
        metrics["ig_completeness_warning_count"] = float(
            warnings.to_numpy().sum()
        )

    for column in sorted(
        column
        for column in slides.columns
        if str(column).startswith("faithfulness_srg/")
    ):
        values = pd.to_numeric(slides[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            suffix = str(column).removeprefix("faithfulness_srg/")
            metrics[f"mean_faithfulness_srg/{suffix}"] = float(finite.mean())
    return metrics


def _report_warnings(
    slide_rows: list[dict[str, Any]],
    metrics: dict[str, float],
) -> list[str]:
    warnings: list[str] = []
    ig_warnings = int(metrics.get("ig_completeness_warning_count", 0))
    if ig_warnings:
        warnings.append(
            f"Integrated-gradients completeness exceeded the configured tolerance "
            f"for {ig_warnings} slide-target result(s); inspect Model diagnostics."
        )
    failed = sum(row.get("status") != "ok" for row in slide_rows)
    unavailable = sum(row.get("summary_status") != "ok" for row in slide_rows)
    if failed:
        warnings.append(
            f"{failed} slide(s) failed attribution; their error status is retained in "
            "the report and tables."
        )
    if unavailable:
        warnings.append(
            f"{unavailable} slide(s) have no static preview, usually because the WSI "
            "was unavailable; raw tile attribution remains available when inference "
            "succeeded."
        )
    return warnings


def _optional_python_value(value: Any) -> Any | None:
    if value is None or bool(pd.isna(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _enabled_methods(config: DictConfig, module: MammaprintModule) -> list[str]:
    methods = []
    if bool(config.leave_one_out.enabled):
        methods.append("leave_one_out")
    if bool(config.ig.enabled):
        methods.append("integrated_gradients")
    if bool(config.single.enabled):
        methods.append("single")
    if bool(config.attention.enabled) and type(module.aggregator).__name__ == "AttentionMIL":
        methods.append("attention")
    return methods


def _as_numpy(value: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _target_scalar(value: Tensor | np.ndarray | float, index: int) -> float:
    values = _as_numpy(value) if not isinstance(value, float) else np.asarray(value)
    return float(values.reshape(-1)[index])


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _slide_seed(seed: int, slide_id: str) -> int:
    """Return a stable CPU-generator seed independent of Python hash randomization."""
    digest = sha256(f"{seed}\0{slide_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


__all__ = [
    "PATHOLOGIST_REVIEW_COLUMNS",
    "SLIDE_KEY_COLUMNS",
    "CohortRunResult",
    "resolve_device",
    "run_cohort",
    "validate_module",
]
