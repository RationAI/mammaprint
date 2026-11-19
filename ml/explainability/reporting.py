"""Adapt explanation tables and masks to RationAI's unmodified report template."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
from report.masks import AbstractMasksRetriever, Masks
from report.reporter import ReportAssembler
from report.save import LocalReportStorer

from ml.explainability.evidence_report import evidence_document, render_evidence
from ml.explainability.viewer import explanation_layers, mlflow_mount_path


if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path


DEFAULT_REPORT_METHODS = ("exact_mean", "leave_one_out", "single", "attention")
REPORT_METHODS = ("integrated_gradients", *DEFAULT_REPORT_METHODS)


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

    def __init__(
        self, *, mlflow_ui_uri: str, included_methods: Collection[str], **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.mlflow_ui_uri = mlflow_ui_uri.rstrip("/")
        self.included_methods = set(included_methods)

    def setup(self) -> None:
        super().setup()
        for retriever in (self.metrics, self.evaluations):
            for metadata in retriever.get_metadata().values():
                # The package defaults to the older MLflow server. Keep the
                # experiment/run suffix and use the UI selected for this job.
                fragment = metadata["mlflow_web_link"].partition("#")[2]
                metadata["mlflow_web_link"] = f"{self.mlflow_ui_uri}/#{fragment}"
        # Keep historical metrics in MLflow, but show only compared methods.
        for kind in ("global_metrics", "metrics"):
            for run_id, frame in self.metrics.data[kind].items():
                self.metrics.data[kind][run_id] = frame.select(
                    [
                        name
                        for name in frame.columns
                        if (name not in REPORT_METHODS or name in self.included_methods)
                        and (
                            not name.startswith("ig_")
                            or "integrated_gradients" in self.included_methods
                        )
                    ]
                )


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
    included_methods: Collection[str] = DEFAULT_REPORT_METHODS,
    evidence_path: Path | None = None,
) -> Path:
    """Render with the upstream assembler; read metadata only, never mask pixels.

    The MLflow slide-key table and metrics must already be logged. Failed slides
    and slides without rendered masks stay in the raw tables, outside the gallery.
    """
    manifest = json.loads(manifest_path.read_text())
    if evidence_path is None and manifest.get("mean_evidence", {}).get("path"):
        evidence_path = manifest_path.parent / "mean_evidence.json"
    evidence = (
        render_evidence(evidence_path)
        if evidence_path is not None and "exact_mean" in included_methods
        else None
    )
    slides = pd.read_parquet(slides_path)
    slides = slides.loc[(slides.status == "ok") & (slides.overlay_status == "ok")]
    if slides.empty:
        if evidence is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(evidence_document(evidence))
            return path
        raise ValueError("No successful slides with rendered masks for the report.")
    if slides.slide_id.duplicated().any():
        raise ValueError("Report slide IDs must be unique.")

    run_url = f"{mlflow_ui_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
    backgrounds = []
    grouped: dict[str, tuple[Any, list[dict[str, str]]]] = {}
    unknown = set(included_methods) - set(REPORT_METHODS)
    if unknown:
        raise ValueError(f"Unknown report methods: {sorted(unknown)}")
    selected = [method for method in included_methods if method in manifest["methods"]]
    if not selected:
        raise ValueError("None of the requested report methods exist in this run.")
    methods = dict.fromkeys(manifest["targets"], selected)
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
        # Removal has no defined empty-bag counterfactual for a one-tile slide.
        slide_methods = {
            target: [
                method
                for method in choices
                if not (method == "leave_one_out" and row.get("tile_count") == 1)
            ]
            for target, choices in methods.items()
        }
        for layer in explanation_layers(row["slide_id"], slide_methods):
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
        included_methods=selected,
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
    if evidence is not None:
        document = path.read_text()
        # Extend the upstream template without forking its gallery or viewer controls.
        position = document.index("</h1>") + len("</h1>")
        path.write_text(document[:position] + evidence + document[position:])
    return path
