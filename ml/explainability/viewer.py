"""Links for reviewing MLflow-hosted explanation masks in xOpat.

The path convention mirrors the RationAI ``report`` package: MLflow artifacts are
mounted in the viewer below ``mflow/<experiment>/<run>/artifacts`` and WSI paths
below ``/mnt`` are exposed relative to that mount.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urljoin, urlparse


if TYPE_CHECKING:
    from collections.abc import Sequence


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
) -> str:
    """Build the same fragment-based xOpat redirect used by RationAI reports."""
    base = _validated_http_base(xopat_ui_uri)
    background_path = _xopat_wsi_path(slide_path)
    data = [background_path]
    shaders: dict[str, dict[str, Any]] = {}
    for index, layer in enumerate(layers, start=1):
        data.append(
            _mlflow_mount_path(
                experiment_id=experiment_id,
                run_id=run_id,
                artifact_root=artifact_root,
                relative_path=layer.artifact_path,
            )
        )
        shaders[f"layer_shader_{index - 1}"] = {
            "name": layer.name,
            "type": "heatmap",
            "dataReferences": [index],
            "fixed": False,
            "visible": bool(layer.visible),
            "params": {
                "use_channel0": "r",
                "color": _validated_hex_color(layer.color),
                "opacity": _validated_opacity(layer.opacity),
            },
        }

    settings = {
        "params": {},
        "data": data,
        "background": [{"dataReference": 0, "lossless": False}],
        "visualizations": [
            {
                "name": str(slide_id),
                "lossless": True,
                "shaders": shaders,
            }
        ],
    }
    payload = json.dumps(settings, separators=(",", ":"), ensure_ascii=False)
    return urljoin(base, "xopat/redirect.php") + "#" + quote(payload, safe="")


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


def _mlflow_mount_path(
    *,
    experiment_id: str,
    run_id: str,
    artifact_root: str,
    relative_path: str,
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
        PurePosixPath("mflow")
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


def _validated_hex_color(color: str) -> str:
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError(f"Invalid viewer color {color!r}.")
    try:
        int(color[1:], 16)
    except ValueError as error:
        raise ValueError(f"Invalid viewer color {color!r}.") from error
    return color


def _validated_opacity(opacity: float) -> float:
    value = float(opacity)
    if not 0 <= value <= 1:
        raise ValueError("Viewer layer opacity must be between 0 and 1.")
    return value


__all__ = ["DEFAULT_XOPAT_UI_URI", "ViewerLayer", "build_xopat_review_url"]
