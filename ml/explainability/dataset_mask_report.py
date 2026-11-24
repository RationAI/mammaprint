"""Add optional dataset-mask layers to a RationAI report's xOpat sessions."""

from __future__ import annotations

import json
import re
from collections import Counter
from html import escape, unescape
from html.parser import HTMLParser
from io import BytesIO
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote, urlsplit

from ml.explainability.tile_removal_report import sha256


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


DEFAULT_XOPAT = "https://xopat.rationai.cloud.trusted.e-infra.cz/"
LAYER_MANIFEST = "dataset-mask-layers.json"
LAYER_ASSETS = "dataset-mask-assets"

# The threshold is on the displayed raster, independently of any tile gate.
_LAYERS = {
    "blur": ("QC blur", "#f7e363"),
    "folding": ("QC folds", "#ff76ce"),
    "residual": ("QC residual artifacts", "#f69c52"),
    "tissue": ("Tissue", "#6be7c1"),
    "epithelium": ("Epithelium", "#6fe0ff"),
    "cancer": ("Cancer probability", "#d397ff"),
}
_IDENTITY = (
    "explanation_run_id",
    "explanation_experiment_id",
    "source_run_id",
    "checkpoint_sha256",
    "split",
    "level",
)
_SHADER_PREFIX = "mammaprint_dataset_mask_"
_HREF_ATTRIBUTES = re.compile(
    r"""\s+([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?"""
)


def _viewer_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or str(path) != value
        or re.search(r"[\\%?#:\x00-\x1f\x7f]", value)
    ):
        raise ValueError("Invalid xOpat mount path.")
    return value


def _wsi_path(value: str) -> str:
    return _viewer_path(value.removeprefix("/mnt/"))


def slide_paths(slides: bytes) -> dict[str, str]:
    """Read the exact WSI roster used in the gallery."""
    import pyarrow.parquet as pq

    rows = pq.read_table(
        BytesIO(slides), columns=["slide_id", "slide_path", "status", "overlay_status"]
    ).to_pylist()
    paths: dict[str, str] = {}
    seen: set[str] = set()
    for row in rows:
        if row["status"] != "ok" or row["overlay_status"] != "ok":
            continue
        slide_id, path = row["slide_id"], row["slide_path"]
        if not isinstance(slide_id, str) or not slide_id.strip() or slide_id in paths:
            raise ValueError("Invalid or repeated slide ID in report roster.")
        if not isinstance(path, str):
            raise ValueError("Missing WSI path in report roster.")
        viewer_path = _wsi_path(path)
        if viewer_path in seen:
            raise ValueError("Ambiguous WSI path in report roster.")
        seen.add(viewer_path)
        paths[slide_id] = path
    if not paths:
        raise ValueError("The report slide roster is empty.")
    return paths


def build_layer_manifest(
    provenance: Mapping[str, Any],
    slides: bytes,
    *,
    dataset_uri: str | None,
    assets: Mapping[str, Any],
    missing: Mapping[str, Any],
    selections: Mapping[str, Any],
    unresolved: str | None = None,
) -> dict[str, Any]:
    """Bind the prepared viewer assets to the exact explanation and roster."""
    return {
        "schema_version": 1,
        "report_identity": {field: provenance.get(field) for field in _IDENTITY},
        "slides_sha256": sha256(slides),
        "dataset_uri": dataset_uri,
        "selections": dict(selections),
        "assets": dict(assets),
        "missing": dict(missing),
        "unresolved": unresolved,
    }


def layer_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def report_xopat_uri(provenance: Mapping[str, Any], explicit: str | None = None) -> str:
    preview = provenance.get("pathologist_preview")
    saved = preview.get("xopat_ui_uri") if isinstance(preview, dict) else None
    return explicit or (saved if isinstance(saved, str) and saved else DEFAULT_XOPAT)


def prepare_layer_manifest(
    client: Any,
    provenance: Mapping[str, Any],
    slides: bytes,
    *,
    artifact_root: str,
    artifact_mount: str,
    resolved_config: bytes | None = None,
    wsi_root: Path | None = None,
    mask_root: Path | None = None,
    export_dir: Path | None = None,
    export_viewer_prefix: str | None = None,
) -> dict[str, Any]:
    if client is None:
        return build_layer_manifest(
            provenance,
            slides,
            dataset_uri=None,
            assets={},
            missing={},
            selections={},
            unresolved="Mask lineage requires access to the source MLflow run.",
        )
    from mlflow.exceptions import MlflowException

    from ml.explainability.dataset_mask_assets import prepare_mask_assets
    from ml.explainability.dataset_mask_lineage import resolve_mask_sources

    try:
        sources = resolve_mask_sources(
            client,
            provenance,
            resolved_config,
            config_artifact=f"{artifact_root}/configs/config-resolved.yaml",
        )
    except MlflowException:
        return build_layer_manifest(
            provenance,
            slides,
            dataset_uri=None,
            assets={},
            missing={},
            selections={},
            unresolved="Could not read dataset lineage metadata from MLflow.",
        )
    assets: dict[str, dict[str, dict[str, str]]] = {}
    missing: dict[str, dict[str, str]] = {}
    if sources.layers:
        assets, missing = prepare_mask_assets(
            client,
            sources.layers,
            slide_paths(slides),
            artifact_mount=artifact_mount,
            wsi_root=wsi_root,
            mask_root=mask_root,
            export_dir=export_dir,
            export_viewer_prefix=export_viewer_prefix,
        )
    return build_layer_manifest(
        provenance,
        slides,
        dataset_uri=sources.dataset_uri,
        assets=assets,
        missing=missing,
        selections={
            kind: {"selection": source.selection, "evidence": source.evidence}
            for kind, source in sources.layers.items()
        },
        unresolved=sources.unresolved,
    )


def _manifest_inputs(
    provenance: Mapping[str, Any], slides: bytes, manifest: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, Any]]:
    identity = {field: provenance.get(field) for field in _IDENTITY}
    run = identity["explanation_run_id"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("report_identity") != identity
        or manifest.get("slides_sha256") != sha256(slides)
        or not isinstance(run, str)
        or not re.fullmatch(r"[a-f0-9]{32}", run)
    ):
        raise ValueError(
            "The mask manifest identity does not match the report and roster."
        )
    paths = slide_paths(slides)
    raw_assets, missing, selections = (
        manifest.get("assets"),
        manifest.get("missing"),
        manifest.get("selections"),
    )
    if not all(isinstance(value, dict) for value in (raw_assets, missing, selections)):
        raise ValueError("Invalid dataset masks in layer manifest.")
    assert isinstance(raw_assets, dict) and isinstance(missing, dict)
    assert isinstance(selections, dict)
    assets: dict[str, dict[str, str]] = {}
    for kind, selection in selections.items():
        if (
            kind not in _LAYERS
            or not isinstance(selection, dict)
            or not all(
                isinstance(selection.get(key), str) and selection[key]
                for key in ("selection", "evidence")
            )
        ):
            raise ValueError("Invalid dataset gate evidence in layer manifest.")
    for slide, items in raw_assets.items():
        if slide not in paths or not isinstance(items, dict):
            raise ValueError("Mask assets must identify slides in the report roster.")
        for kind, item in items.items():
            if (
                kind not in selections
                or not isinstance(item, dict)
                or not isinstance(item.get("source"), str)
                or not item["source"]
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
                or not isinstance(item.get("viewer_path"), str)
                or PurePosixPath(item["viewer_path"]).suffix.lower()
                not in {".tif", ".tiff"}
            ):
                raise ValueError("Invalid viewer asset in dataset mask manifest.")
            assets.setdefault(slide, {})[kind] = _viewer_path(item["viewer_path"])
    for slide, items in missing.items():
        if (
            slide not in paths
            or not isinstance(items, dict)
            or any(
                kind not in _LAYERS
                or not isinstance(reason, str)
                or not reason
                or kind in assets.get(slide, {})
                for kind, reason in items.items()
            )
        ):
            raise ValueError("Invalid missing-mask records in layer manifest.")
    return paths, assets, missing


def _shader(kind: str, index: int) -> dict[str, Any]:
    title, color = _LAYERS[kind]
    # The zero cutoff masks background. Cancer retains its source byte scale.
    return {
        "name": title,
        "type": "heatmap",
        "dataReferences": [index],
        "fixed": False,
        "visible": False,
        "color": color,
        "opacity": 0.45,
        "params": {
            "use_channel0": "r",
            "color": color,
            "opacity": 0.45,
            "threshold": {
                "type": "range_input",
                "default": 0,
                "min": 0,
                "max": 100,
                "step": 1,
                "title": "Display cutoff (% of scale): ",
            },
        },
    }


class _GalleryLinks(HTMLParser):
    def __init__(
        self,
        source: str,
        viewer: str,
        run_prefix: re.Pattern[str],
        paths: dict[str, str],
        assets: dict[str, dict[str, str]],
    ):
        super().__init__(convert_charrefs=True)
        self.offsets = [0, *(match.end() for match in re.finditer("\n", source))]
        self.viewer = urlsplit(viewer)
        if (
            self.viewer.scheme not in {"https", "http"}
            or not self.viewer.hostname
            or (
                self.viewer.username
                or self.viewer.password
                or self.viewer.query
                or self.viewer.fragment
            )
        ):
            raise ValueError("Expected an absolute xOpat base URL.")
        self.redirect_path = self.viewer.path.rstrip("/") + "/xopat/redirect.php"
        self.run_prefix = run_prefix
        self.slides = {_wsi_path(path): slide_id for slide_id, path in paths.items()}
        self.assets = assets
        self.roots: set[str] = set()
        self.matched: Counter[str] = Counter()
        self.edits: list[tuple[int, int, str]] = []
        self.feed(source)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        urls = [value for name, value in attrs if name == "href" and value]
        if len(urls) != 1:
            return
        href = urls[0]
        url = urlsplit(href)
        if (url.scheme, url.netloc, url.path) != (
            self.viewer.scheme,
            self.viewer.netloc,
            self.redirect_path,
        ) or not url.fragment:
            return
        try:
            session = json.loads(unquote(url.fragment))
        except (ValueError, UnicodeError):
            return
        if not isinstance(session, dict) or not isinstance(
            data := session.get("data"), list
        ):
            return
        roots = {
            match.group()
            for path in data
            if isinstance(path, str) and (match := self.run_prefix.match(path))
        }
        if not roots:
            return
        self.roots.update(roots)

        try:
            backgrounds = session["background"]
            reference = backgrounds[0]["dataReference"]
            if type(reference) is not int or not 0 <= reference < len(data):
                raise ValueError("Invalid WSI data reference.")
            slide = self.slides[data[reference]]
            shaders = session["visualizations"][0]["shaders"]
            if not isinstance(shaders, dict) or not any(
                key.startswith("layer_shader_") for key in shaders
            ):
                raise ValueError("Unrecognized report shader keys.")
        except (KeyError, TypeError, IndexError, ValueError) as error:
            raise ValueError(
                "A report gallery link does not match the slide roster or session format."
            ) from error
        self.matched[slide] += 1

        desired = self.assets.get(slide, {})
        managed = {
            key.removeprefix(_SHADER_PREFIX)
            for key in shaders
            if key.startswith(_SHADER_PREFIX)
        }
        if managed - desired.keys():
            raise ValueError(
                "Existing dataset layers differ from this manifest. Restore the original report first."
            )
        changed = False
        for kind in _LAYERS:
            if kind not in desired:
                continue
            path = desired[kind]
            if path not in data:
                data.append(path)
            shader = _shader(kind, data.index(path))
            key = _SHADER_PREFIX + kind
            if key in shaders:
                if shaders[key] != shader:
                    raise ValueError(
                        "Existing dataset layers differ from this manifest. Restore the original report first."
                    )
                continue
            shaders[key] = shader
            changed = True
        if changed:
            fragment = quote(
                json.dumps(session, ensure_ascii=False, separators=(",", ":")), safe=""
            )
            self._edit_href(href, url._replace(fragment=fragment).geturl())

    def _edit_href(self, href: str, updated: str) -> None:
        raw = self.get_starttag_text()
        assert raw is not None
        for attr in _HREF_ATTRIBUTES.finditer(raw):
            if attr.group(1).lower() != "href":
                continue
            index = next((i for i in (2, 3, 4) if attr.group(i) is not None), None)
            if index is None or unescape(attr.group(index)) != href:
                continue
            line, column = self.getpos()
            start = self.offsets[line - 1] + column
            replacement = escape(updated, quote=True)
            if index == 4:
                replacement = f'"{replacement}"'
            self.edits.append(
                (start + attr.start(index), start + attr.end(index), replacement)
            )
            return
        raise ValueError("Could not locate the report gallery href safely.")


def patch_report_layers(
    original: bytes,
    provenance: Mapping[str, Any],
    slides: bytes,
    manifest: Mapping[str, Any],
    *,
    xopat_ui_uri: str,
    artifact_root: str,
    artifact_mount: str,
) -> tuple[bytes, dict[str, Any]]:
    paths, assets, missing = _manifest_inputs(provenance, slides, manifest)
    summary = {
        "gallery_slides": len(paths),
        "slides_with_masks": len(assets),
        "layers_by_type": dict(
            Counter(kind for items in assets.values() for kind in items)
        ),
        "missing_masks": dict(
            Counter(kind for items in missing.values() for kind in items)
        ),
        "unresolved": manifest.get("unresolved"),
        "patched_links": 0,
    }
    if not assets:
        if _SHADER_PREFIX.encode() in original:
            raise ValueError(
                "The report already has dataset layers. Provide its layer manifest."
            )
        return original, summary
    source = original.decode()
    mount = _viewer_path(artifact_mount)
    root = _viewer_path(artifact_root)
    run = provenance["explanation_run_id"]
    experiment = provenance.get("explanation_experiment_id")
    if experiment is not None and not re.fullmatch(r"[0-9]+", str(experiment)):
        raise ValueError("Invalid explanation experiment in report provenance.")
    pattern = re.compile(
        rf"{re.escape(mount)}/{experiment if experiment is not None else '[0-9]+'}"
        rf"/{run}/artifacts/{re.escape(root)}/masks/"
    )
    parser = _GalleryLinks(source, xopat_ui_uri, pattern, paths, assets)
    if len(parser.roots) != 1 or parser.matched != Counter(dict.fromkeys(paths, 2)):
        raise ValueError("The xOpat gallery does not match the report slide roster.")
    parts: list[str] = []
    end = 0
    for start, stop, replacement in parser.edits:
        parts.extend((source[end:start], replacement))
        end = stop
    parts.append(source[end:])
    summary["patched_links"] = len(parser.edits)
    return "".join(parts).encode(), summary
