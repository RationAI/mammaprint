"""Adapt explanation tables and masks to RationAI's unmodified report template."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
from report.masks import AbstractMasksRetriever, Masks
from report.reporter import ReportAssembler
from report.save import LocalReportStorer

from ml.explainability.viewer import explanation_layers, mlflow_mount_path


if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path


class _MappedMasks(AbstractMasksRetriever):
    """Keep explicit slide IDs when matching our signed OME-TIFF filenames."""

    def __init__(
        self,
        name: str,
        rows: list[dict[str, Any]],
        *,
        source: str,
        run_url: str | None,
        shader_conf: dict[str, Any] | None = None,
    ):
        super().__init__(layer_name=name, shader_conf=shader_conf)
        self.rows = rows
        self.metadata = self._compose_metadata(run_url, source)

    def get_masks(self) -> Masks:
        return pl.from_dicts(
            [
                {**row, "shader_conf": self.shader_conf_map(row["mask_path"])}
                for row in self.rows
            ]
        ), self.metadata


class _ReportAssembler(ReportAssembler):
    """Use this deployment's MLflow UI in the standard report's run links."""

    def __init__(self, *, mlflow_ui_uri: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.mlflow_ui_uri = mlflow_ui_uri.rstrip("/")

    def setup(self) -> None:
        super().setup()
        for retriever in (self.metrics, self.evaluations):
            for metadata in retriever.get_metadata().values():
                # The package defaults to the older MLflow server. Keep the
                # experiment/run suffix and use the UI selected for this job.
                fragment = metadata["mlflow_web_link"].partition("#")[2]
                metadata["mlflow_web_link"] = f"{self.mlflow_ui_uri}/#{fragment}"


def write_rationai_report(
    path: Path,
    *,
    slides_path: Path,
    manifest_path: Path,
    run_id: str,
    experiment_id: str,
    artifact_root: str,
    slide_key_artifact: str,
    xopat_ui_uri: str,
    mlflow_ui_uri: str,
    artifact_mount: str = "secure_mlflow",
    mask_artifact_paths: Collection[str],
) -> Path:
    """Render with the upstream assembler; read metadata only, never mask pixels.

    The MLflow slide-key table and metrics must already be logged. Failed slides
    and slides without rendered masks stay in the raw tables, outside the gallery.
    """
    manifest = json.loads(manifest_path.read_text())
    slides = pd.read_parquet(slides_path)
    slides = slides.loc[(slides.status == "ok") & (slides.overlay_status == "ok")]
    if slides.empty:
        raise ValueError("No successful slides with rendered masks for the report.")
    if slides.slide_id.duplicated().any():
        raise ValueError("Report slide IDs must be unique.")

    run_url = f"{mlflow_ui_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
    backgrounds = []
    grouped: dict[str, tuple[Any, list[dict[str, str]]]] = {}
    methods = dict.fromkeys(manifest["targets"], manifest["methods"])
    available = set(mask_artifact_paths)
    for row in slides.to_dict("records"):
        truth = row.get("mammaprint_index")
        label = (
            f"MammaPrint index: {float(truth):.4f}"
            if pd.notna(truth)
            else row.get("class")
        )
        backgrounds.append(
            {
                "item_name": row["slide_id"],
                "mask_path": row["slide_path"],
                "ground_truth": label,
            }
        )
        for layer in explanation_layers(row["slide_id"], methods):
            artifact_path = f"{artifact_root.rstrip('/')}/{layer.artifact_path}"
            if artifact_path not in available:
                raise ValueError(f"Missing mask artifact: {artifact_path}")
            if layer.name not in grouped:
                grouped[layer.name] = (layer, [])
            grouped[layer.name][1].append(
                {
                    "item_name": row["slide_id"],
                    "mask_path": mlflow_mount_path(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        artifact_root=artifact_root,
                        relative_path=layer.artifact_path,
                        artifact_mount=artifact_mount,
                    ),
                }
            )

    assembler = _ReportAssembler(
        mlflow_ui_uri=mlflow_ui_uri,
        title="MammaPrint tile explanations",
        background=_MappedMasks(
            "Original WSI", backgrounds, source="drive", run_url=run_url
        ),
        save=LocalReportStorer(str(path.resolve())),
        template_asset_path="report_template.html",
        metrics_run_ids=[run_id],
        evaluation_runs={
            run_id: {"artifact_file": slide_key_artifact, "slide_item_key": "slide_id"}
        },
        mask_retrievers=[
            _MappedMasks(
                name,
                rows,
                source="mlflow",
                run_url=run_url,
                shader_conf={
                    "type": "heatmap",
                    "color": layer.color,
                    "opacity": layer.opacity,
                    "visible": layer.visible,
                },
            )
            for name, (layer, rows) in grouped.items()
        ],
    )
    assembler.xopat_base_url = xopat_ui_uri.rstrip("/") + "/"
    assembler.assemble_report()
    if run_id not in assembler.evaluations.get_metadata():
        path.unlink(missing_ok=True)
        raise RuntimeError("RationAI report could not load the per-slide MLflow table.")
    return path
