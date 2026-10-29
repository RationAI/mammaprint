"""Links for reviewing MLflow-hosted explanation masks in xOpat.

The path convention mirrors the RationAI ``report`` package: MLflow artifacts are
mounted in the viewer below ``secure_mlflow/<experiment>/<run>/artifacts`` and WSI paths
below ``/mnt`` are exposed relative to that mount.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urljoin, urlparse


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence


DEFAULT_XOPAT_UI_URI = "https://xopat.rationai.cloud.trusted.e-infra.cz/"


@dataclass(frozen=True, slots=True)
class ViewerLayer:
    """One scalar TIFF layer in the interactive slide viewer."""

    name: str
    artifact_path: str
    color: str
    visible: bool = False
    opacity: float = 0.7


def build_xopat_review_url(
    *,
    xopat_ui_uri: str,
    experiment_id: str,
    run_id: str,
    artifact_root: str,
    slide_id: str,
    slide_path: str,
    layers: Sequence[ViewerLayer],
    artifact_mount: str = "secure_mlflow",
) -> str:
    """Delegate viewer settings and links to the RationAI reporting package."""
    from report.xopat_helpers import (
        assemble_xopat_redirect_url,
        setup_item_visualisation_setting,
    )

    layer_data = [
        {
            "path": mlflow_mount_path(
                experiment_id=experiment_id,
                run_id=run_id,
                artifact_root=artifact_root,
                relative_path=layer.artifact_path,
                artifact_mount=artifact_mount,
            ),
            "shader_conf": {
                "type": "heatmap",
                "visible": layer.visible,
                "color": layer.color,
                "opacity": layer.opacity,
            },
            "metadata": {"layer_name": layer.name},
        }
        for layer in layers
    ]
    settings = setup_item_visualisation_setting(
        slide_id, _xopat_wsi_path(slide_path), layer_data
    )
    return assemble_xopat_redirect_url(settings, _validated_http_base(xopat_ui_uri))


def build_xopat_report_url(
    *,
    xopat_ui_uri: str,
    experiment_id: str,
    run_id: str,
    artifact_path: str,
    artifact_mount: str = "secure_mlflow",
) -> str:
    path = mlflow_mount_path(
        experiment_id=experiment_id,
        run_id=run_id,
        artifact_root=artifact_path,
        relative_path="report.html",
        artifact_mount=artifact_mount,
    )
    return (
        urljoin(_validated_http_base(xopat_ui_uri), "index.php")
        + "?"
        + urlencode({"p": str(PurePosixPath(path).parent), "view": "report.html"})
    )


def _xopat_wsi_path(slide_path: str) -> str:
    path = PurePosixPath(slide_path)
    parts = path.parts
    if path.is_absolute():
        if len(parts) < 2 or parts[1] != "mnt":
            raise ValueError(
                "xOpat can only address absolute WSI paths mounted below /mnt; "
                f"got {slide_path!r}."
            )
        path = PurePosixPath(*parts[2:])
    return _safe_relative_path(path, field="slide_path")


def mlflow_mount_path(
    *,
    experiment_id: str,
    run_id: str,
    artifact_root: str,
    relative_path: str,
    artifact_mount: str = "secure_mlflow",
) -> str:
    components = (
        _safe_component(experiment_id, "experiment_id"),
        _safe_component(run_id, "run_id"),
    )
    root = _safe_relative_path(PurePosixPath(artifact_root), field="artifact_root")
    relative = _safe_relative_path(
        PurePosixPath(relative_path), field="layer artifact_path"
    )
    return str(
        PurePosixPath(_safe_component(artifact_mount, "artifact_mount"))
        / components[0]
        / components[1]
        / "artifacts"
        / root
        / relative
    )


def _safe_component(value: str, field: str) -> str:
    if not value or "/" in value or value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty path component.")
    return value


def _safe_relative_path(path: PurePosixPath, *, field: str) -> str:
    if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
        raise ValueError(f"{field} must be a safe relative path; got {str(path)!r}.")
    value = path.as_posix()
    if value in {"", "."}:
        raise ValueError(f"{field} must not be empty.")
    return value


def _validated_http_base(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("xopat_ui_uri must be an absolute HTTP(S) URL.")
    return uri.rstrip("/") + "/"


def explanation_layers(
    slide_id: str,
    targets: Mapping[str, Collection[str]],
) -> list[ViewerLayer]:
    """Describe every generated TIFF using clinically honest layer names."""
    layers: list[ViewerLayer] = []
    first_target = next(iter(targets), None)
    attention_added = False
    colors = {
        "integrated_gradients": ("#d73027", "#4575b4"),
        "leave_one_out": ("#ef476f", "#118ab2"),
        "single": ("#f28e2b", "#9467bd"),
    }
    for target, methods in targets.items():
        target_label = {
            "luminal_a_logit": "Luminal A evidence",
            "mammaprint_index": "MammaPrint index",
        }.get(target, target.replace("_", " "))
        initial_method = next(
            (
                method
                for method in (
                    "integrated_gradients",
                    "leave_one_out",
                    "single",
                    "attention",
                )
                if method in methods
            ),
            None,
        )
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
            initially_visible = method == initial_method and target == first_target
            layers.extend(
                (
                    ViewerLayer(
                        name=f"{method_label} — {directions[0]} {target_label}",
                        artifact_path=(f"{mask_root}/{slide_id}_positive.ome.tif"),
                        color=positive_color,
                        visible=initially_visible,
                    ),
                    ViewerLayer(
                        name=f"{method_label} — {directions[1]} {target_label}",
                        artifact_path=(f"{mask_root}/{slide_id}_negative.ome.tif"),
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
                        f"masks/attention/{target}/{slide_id}_positive.ome.tif"
                    ),
                    color="#f2c14e",
                    visible=initial_method == "attention" and target == first_target,
                )
            )
            attention_added = True
    return layers
