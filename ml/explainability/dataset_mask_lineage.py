"""Find mask sources and tile gates in the dataset that supplied a report."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

import yaml


if TYPE_CHECKING:
    from collections.abc import Mapping


_RESOLVED_CONFIG = "configs/config-resolved.yaml"
_RESOLUTION_INVENTORY = "metadata/l1_inventory.json"
_RUN_ID = re.compile(r"[a-f0-9]{32}")
_ARTIFACT_URI = re.compile(r"mlflow-artifacts:/([0-9]+)/([a-f0-9]{32})/artifacts/(.+)")
_CRITICAL_CONFIG_PATHS = (
    "seed",
    "label_mode",
    "feature_dim",
    "dataset",
    "datamodule",
    "ml._target_",
    "ml.encoder",
    "ml.aggregator",
    "ml.head",
    "ml.output_activation",
)
_GATE_LABELS = {
    "blur": "blur mask clean fraction",
    "folding": "folding mask clean fraction",
    "residual": "residual artifact mask clean fraction",
    "tissue": "tissue overlap in the central half of the tile",
    "epithelium": "epithelium overlap",
    "cancer": "cancer overlap",
}


class ArtifactReader(Protocol):
    def get_run(self, run_id: str) -> Any: ...

    def list_artifacts(self, run_id: str, path: str | None = None) -> list[Any]: ...

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str: ...


@dataclass(frozen=True)
class MaskSource:
    location: str
    selection: str
    evidence: str


@dataclass(frozen=True)
class MaskSources:
    dataset_uri: str | None
    layers: dict[str, MaskSource]
    unresolved: str | None = None


@dataclass(frozen=True)
class _Artifact:
    experiment: str
    run: str
    path: str

    def uri(self, path: str) -> str:
        return f"mlflow-artifacts:/{self.experiment}/{self.run}/artifacts/{path}"


class _LineageUnavailableError(ValueError):
    """The recorded evidence is missing, inconsistent, or unsupported."""


def _safe_artifact_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        re.fullmatch(r"[a-zA-Z0-9._/-]+", value) is not None
        and not path.is_absolute()
        and bool(path.parts)
        and ".." not in path.parts
        and path.as_posix() == value
    )


def _artifact(value: Any) -> _Artifact | None:
    match = _ARTIFACT_URI.fullmatch(value) if isinstance(value, str) else None
    if match is None or not _safe_artifact_path(match[3]):
        return None
    return _Artifact(*match.groups())


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _lookup(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = _mapping(value).get(part)
    return value


def _hash(config: dict[str, Any]) -> str:
    try:
        payload = json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    except (TypeError, ValueError):
        raise _LineageUnavailableError(
            "Cannot hash the saved lineage config."
        ) from None
    return hashlib.sha256(payload).hexdigest()


def _download(client: ArtifactReader, run: str, path: str) -> bytes | None:
    if not _safe_artifact_path(path) or not _RUN_ID.fullmatch(run):
        raise _LineageUnavailableError("Invalid MLflow metadata pointer.")
    parent = PurePosixPath(path).parent
    listing = None if parent == PurePosixPath(".") else parent.as_posix()
    try:
        files = client.list_artifacts(run, listing)
    except FileNotFoundError:
        return None
    except Exception as error:
        if getattr(error, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    if not any(item.path == path and not item.is_dir for item in files):
        return None
    with tempfile.TemporaryDirectory() as directory:
        file = Path(client.download_artifacts(run, path, directory))
        return file.read_bytes()


def _yaml(data: bytes | None) -> dict[str, Any]:
    if data is None:
        return {}
    try:
        config = _mapping(yaml.safe_load(data))
    except yaml.YAMLError:
        raise _LineageUnavailableError(
            "Invalid YAML in a saved lineage config."
        ) from None
    if not config:
        raise _LineageUnavailableError("Saved lineage config is empty or invalid.")
    return config


def _report_config(
    client: ArtifactReader,
    provenance: Mapping[str, Any],
    supplied: bytes | None,
    config_artifact: str,
) -> dict[str, Any]:
    run = provenance.get("explanation_run_id")
    expected = provenance.get("config_hash")
    if supplied is not None and expected is None:
        raise _LineageUnavailableError(
            "A supplied report config requires a hash in the explanation manifest."
        )
    if (
        supplied is None
        and expected
        and isinstance(run, str)
        and _RUN_ID.fullmatch(run)
    ):
        supplied = _download(client, run, config_artifact)
    if supplied is not None:
        config = _yaml(supplied)
        if _hash(config) != expected:
            raise _LineageUnavailableError(
                "Saved explanation config hash differs from its manifest."
            )
        return config

    source = provenance.get("source_run_id")
    source_artifact = provenance.get("source_config_artifact")
    raw_hash = provenance.get("source_config_sha256")
    critical_hash = provenance.get("source_critical_config_hash")
    if (
        not isinstance(source, str)
        or not isinstance(source_artifact, str)
        or not isinstance(raw_hash, str)
        or not isinstance(critical_hash, str)
    ):
        raise _LineageUnavailableError(
            "No saved explanation config or verified training-config fallback."
        )
    if source == run:
        raise _LineageUnavailableError(
            "The training-config fallback identifies the explanation run."
        )
    raw = _download(client, source, source_artifact)
    if raw is None or hashlib.sha256(raw).hexdigest() != raw_hash:
        raise _LineageUnavailableError(
            "Cannot verify the source training config digest."
        )
    config = _yaml(raw)
    critical = {path: _lookup(config, path) for path in _CRITICAL_CONFIG_PATHS}
    if None in critical.values() or _hash(critical) != critical_hash:
        raise _LineageUnavailableError(
            "Cannot verify the source training config against the explanation."
        )
    return config


def _select_card(config: dict[str, Any], split: str, level: int) -> dict[str, Any]:
    dataset = _mapping(_lookup(config, f"datamodule.{split}"))
    if dataset.get("split", split) != split:
        raise _LineageUnavailableError(
            "The resolved dataset split differs from the explanation split."
        )
    levels = _mapping(dataset.get("levels"))
    if len(levels) != 1 or str(next(iter(levels))) != str(level):
        raise _LineageUnavailableError(
            "The resolved dataset level differs from the explanation level."
        )
    return _mapping(next(iter(levels.values())))


def _run(client: ArtifactReader, artifact: _Artifact) -> Any:
    try:
        run = client.get_run(artifact.run)
    except Exception as error:
        if (
            not isinstance(error, FileNotFoundError)
            and getattr(error, "error_code", None) != "RESOURCE_DOES_NOT_EXIST"
        ):
            raise
        raise _LineageUnavailableError(
            "A dataset lineage run is unavailable."
        ) from None
    if (
        run.info.run_id != artifact.run
        or str(run.info.experiment_id) != artifact.experiment
        or run.info.lifecycle_stage != "active"
    ):
        raise _LineageUnavailableError(
            "The selected dataset artifact does not match an active MLflow run."
        )
    return run


def _threshold(value: Any) -> str:
    if (
        type(value) not in (int, float)
        or not 0 <= value <= 1
        or not math.isfinite(value)
    ):
        raise _LineageUnavailableError(
            "A recorded tile-selection threshold is invalid."
        )
    return str(value)


def _filter_selection(data: bytes, level: int) -> tuple[_Artifact, dict[str, str]]:
    try:
        manifest = _mapping(json.loads(data))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _LineageUnavailableError("Invalid dataset filter manifest.") from None
    parent = _artifact(manifest.get("source_tiling_uri"))
    strategy = manifest.get("strategy")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("level") != level
        or manifest.get("comparison") != ">"
        or strategy not in ("and", "or", "epithelium_only")
        or parent is None
        or _artifact(manifest.get("source_embeddings_uri")) is None
    ):
        raise _LineageUnavailableError(
            "The dataset filter manifest has unsupported or mismatched lineage."
        )
    epi = f"epithelium overlap > {_threshold(manifest.get('epithelium_threshold'))}"
    if strategy == "epithelium_only":
        return parent, {"epithelium": epi}
    cancer = f"cancer overlap > {_threshold(manifest.get('cancer_threshold'))}"
    rule = f"{epi} {strategy.upper()} {cancer}"
    return parent, dict.fromkeys(("epithelium", "cancer"), rule)


def _resolution_tiling(
    client: ArtifactReader,
    artifact: _Artifact,
    split: str,
    recorded_digest: Any,
) -> _Artifact:
    data = _download(client, artifact.run, _RESOLUTION_INVENTORY)
    if data is None:
        raise _LineageUnavailableError("The registered L1 inventory is missing.")
    try:
        inventory = _mapping(json.loads(data))
        selection = _mapping(inventory.get("source_selection"))
        contents = {key: value for key, value in selection.items() if key != "digest"}
        # The selection writer uses JSON's default separators and ASCII escaping.
        payload = json.dumps(contents, sort_keys=True, allow_nan=False).encode()
    except ValueError:
        raise _LineageUnavailableError("Invalid L1 dataset inventory.") from None
    digest = hashlib.sha256(payload).hexdigest()
    tiling = _artifact(selection.get("tiling_source"))
    if (
        selection.get("level") != 1
        or selection.get("schema_version") != 1
        or selection.get("digest") != digest
        or inventory.get("source_manifest_digest") != digest
        or _lookup(inventory, "embedding_config.manifest_digest") != digest
        or recorded_digest != digest
        or not _mapping(_lookup(inventory, f"successful.{split}"))
        or tiling is None
    ):
        raise _LineageUnavailableError(
            "The L1 dataset inventory has mismatched lineage."
        )
    return tiling


def _tiling_sources(
    config: dict[str, Any],
    tiling: _Artifact,
    filter_rules: dict[str, str],
    filter_evidence: str | None,
    inventory_evidence: str | None,
) -> tuple[dict[str, MaskSource], list[str]]:
    dataset = _mapping(config.get("dataset"))
    paths = _mapping(dataset.get("paths"))
    uris = _mapping(dataset.get("mlflow_uris"))
    locations = {
        "blur": paths.get("qc_masks"),
        "folding": paths.get("qc_masks"),
        "residual": paths.get("qc_masks"),
        "tissue": uris.get("tissue_masks"),
        "epithelium": paths.get("epithelium_masks"),
        "cancer": uris.get("cancer_masks"),
    }
    sources = {}
    missing = []
    config_evidence = tiling.uri(_RESOLVED_CONFIG)
    for name, location in locations.items():
        cutoff = config.get(f"{name}_threshold")
        rule = filter_rules.get(name)
        if name in {"epithelium", "cancer"} and location in (None, ""):
            cutoff = None  # The tiler skips these optional masks when the source is disabled.
        if cutoff is None and rule is None:
            continue
        if not _valid_location(location):
            missing.append(name)
            continue
        try:
            tiling_rule = (
                f"{_GATE_LABELS[name]} > {_threshold(cutoff)}"
                if cutoff is not None
                else None
            )
        except _LineageUnavailableError:
            missing.append(name)
            continue
        selection = (
            f"Tiling requires {tiling_rule}. Embedding filter requires {rule}."
            if tiling_rule and rule
            else tiling_rule or rule
        )
        assert selection is not None
        evidence = ", ".join(
            value
            for value in (
                config_evidence,
                inventory_evidence,
                filter_evidence if rule else None,
            )
            if value
        )
        sources[name] = MaskSource(location, selection, evidence)
    return sources, missing


def _valid_location(value: Any) -> TypeGuard[str]:
    if _artifact(value) is not None:
        return True
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value.startswith("//")
    ):
        return False
    path = PurePosixPath(value)
    return (
        len(path.parts) > 1
        and path.as_posix() == value
        and ".." not in path.parts
        and not any(char in value for char in "\\:$\x00\n\r")
    )


def _check_embedding_split(
    client: ArtifactReader, split: _Artifact, whole: _Artifact
) -> None:
    def parquet_files(artifact: _Artifact) -> dict[str, int]:
        files = {}
        for item in client.list_artifacts(artifact.run, artifact.path):
            path = PurePosixPath(item.path)
            if not item.is_dir and path.suffix == ".parquet":
                if (
                    path.parent.as_posix() != artifact.path
                    or type(item.file_size) is not int
                    or item.file_size <= 0
                ):
                    raise _LineageUnavailableError(
                        "The registered embedding artifact inventory is incomplete."
                    )
                files[path.name] = item.file_size
        return files

    _run(client, whole)
    split_files = parquet_files(split)
    whole_files = parquet_files(whole)
    if not split_files or not split_files.items() <= whole_files.items():
        raise _LineageUnavailableError(
            "The historical split does not match the registered whole-dataset inventory."
        )


def _trace(
    client: ArtifactReader,
    selected: _Artifact,
    whole: _Artifact | None,
    split: str,
    level: int,
    card_provenance: Any,
) -> tuple[dict[str, MaskSource], list[str]]:
    artifact = selected
    visited: set[_Artifact] = set()
    filter_rules: dict[str, str] = {}
    filter_evidence = None
    inventory_evidence = None
    split_evidence = None
    for _ in range(5):
        if artifact in visited:
            raise _LineageUnavailableError(
                "Dataset preprocessing lineage contains a cycle."
            )
        visited.add(artifact)
        run = _run(client, artifact)
        if not filter_rules and artifact.path in {"embeddings", f"embeddings_{split}"}:
            manifest = _download(client, artifact.run, "filter_manifest.json")
            if manifest is not None:
                filter_evidence = artifact.uri("filter_manifest.json")
                artifact, filter_rules = _filter_selection(manifest, level)
                continue
        resolution_digest = run.data.tags.get("resolution_manifest")
        if (
            level == 1
            and artifact == selected
            and artifact.path == f"embeddings_{split}"
            and (resolution_digest is not None or card_provenance is not None)
        ):
            inventory_evidence = artifact.uri(_RESOLUTION_INVENTORY)
            if card_provenance is not None and card_provenance != inventory_evidence:
                raise _LineageUnavailableError(
                    "The L1 data card points to an inventory outside the selected run."
                )
            artifact = _resolution_tiling(client, artifact, split, resolution_digest)
            continue
        config = _yaml(_download(client, artifact.run, _RESOLVED_CONFIG))
        if config.get("data_name") == artifact.path:
            if str(config.get("level", level)) != str(level):
                raise _LineageUnavailableError(
                    "The source tiling config has the wrong level."
                )
            return _tiling_sources(
                config,
                artifact,
                filter_rules,
                filter_evidence,
                inventory_evidence or split_evidence,
            )
        embedding_parent = _artifact(_lookup(config, "dataset.path"))
        if artifact.path == "embeddings" and embedding_parent is not None:
            artifact = embedding_parent
            continue
        if (
            artifact == selected
            and whole is not None
            and whole.path == "embeddings"
            and artifact.path.endswith(f"_{split}")
            and run.data.tags.get("mlflow.runName") == f"split-{artifact.path}"
        ):
            _check_embedding_split(client, artifact, whole)
            split_evidence = f"{artifact.uri(artifact.path)} matched by filenames and sizes to {whole.uri(whole.path)}"
            artifact = whole
            continue
        break
    raise _LineageUnavailableError(
        "No verified tiling lineage for the selected dataset artifact."
    )


def resolve_mask_sources(
    client: ArtifactReader,
    provenance: Mapping[str, Any],
    resolved_config: bytes | None,
    *,
    config_artifact: str = f"explanations/{_RESOLVED_CONFIG}",
) -> MaskSources:
    """Resolve the selected split and level without inferring gates from dataset names."""
    dataset_uri = None
    try:
        split, level_value = provenance.get("split"), str(provenance.get("level"))
        if split not in ("train", "val", "test") or not re.fullmatch(
            r"[0-9]+", level_value
        ):
            raise _LineageUnavailableError(
                "Explanation split or level metadata is missing."
            )
        level = int(level_value)
        config = _report_config(client, provenance, resolved_config, config_artifact)
        card = _select_card(config, split, level)
        uris = _mapping(card.get("uris"))
        selected = _artifact(uris.get(split))
        if selected is None:
            raise _LineageUnavailableError(
                "The selected split does not identify a supported MLflow artifact."
            )
        dataset_uri = uris[split]
        layers, missing = _trace(
            client,
            selected,
            _artifact(uris.get("all")),
            split,
            level,
            card.get("provenance"),
        )
    except _LineageUnavailableError as error:
        return MaskSources(dataset_uri, {}, str(error))
    if missing:
        return MaskSources(
            dataset_uri, layers, "Unverified mask sources: " + ", ".join(missing)
        )
    return MaskSources(dataset_uri, layers)
