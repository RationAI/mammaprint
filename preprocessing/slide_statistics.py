"""Build a versioned cache of slide summaries from existing embedding Parquets."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import heapq
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


SCHEMA_VERSION = 1
FEATURES = ("mean_raw", "std_raw", "mean_ln", "std_ln")
SPLITS = ("train", "val", "test")
SOURCE_KEY = re.compile(r"l[1-9][0-9]*/[a-z][a-z0-9_]*")
ENV_URI = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}")
COVERAGE_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("slide_id", pa.string()),
        ("split", pa.string()),
        ("status", pa.string()),
        ("tile_count", pa.int64()),
        ("reason", pa.string()),
    ]
)


@dataclass(frozen=True)
class Slide:
    slide_id: str
    record_num: str
    split: str
    target: float | None


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _label(row: Mapping[str, str]) -> float | None:
    try:
        value = float(row["mammaprint_index"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_membership(mapping: Path, *, labels: str | None = None) -> dict[str, Slide]:
    """Read membership and interpret labels only for the requested partition."""
    if labels not in (None, "dev", "test"):
        raise ValueError(f"Unknown label partition {labels!r}")
    required = {"path", "split", "record_num"}
    if labels is not None:
        required.add("mammaprint_index")
    slides: dict[str, Slide] = {}
    with mapping.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if missing := required - set(rows.fieldnames or ()):
            raise ValueError(f"Mapping lacks columns {sorted(missing)}")
        for row in rows:
            path = row["path"]
            if not path or not path.strip():
                continue
            slide_id = Path(path).stem
            record = (row["record_num"] or "").strip()
            split = row["split"]
            if not slide_id or slide_id in slides:
                raise ValueError(f"Empty or duplicate mapping slide stem {slide_id!r}")
            if split not in SPLITS or not record:
                raise ValueError(
                    f"Invalid split or missing record_num for slide {slide_id!r}"
                )
            partition = "test" if split == "test" else "dev"
            target = _label(row) if labels == partition else None
            slides[slide_id] = Slide(slide_id, record, split, target)
    if not slides:
        raise ValueError("Mapping contains no slide IDs")
    test_records = {
        slide.record_num for slide in slides.values() if slide.split == "test"
    }
    dev_records = {
        slide.record_num for slide in slides.values() if slide.split != "test"
    }
    if test_records & dev_records:
        raise ValueError(
            f"{len(test_records & dev_records)} record_num values span development and test"
        )
    return slides


def source_inventory(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve the configured cards without contacting MLflow or opening embeddings."""
    entries: dict[str, dict[str, Any]] = {}
    if cards_dir := config.get("cards_dir"):
        root = Path(cards_dir)
        if not root.is_dir():
            raise ValueError(f"Embedding cards directory does not exist: {root}")
        for card in sorted(root.glob("*/l*.yaml")):
            key = f"{card.stem}/{card.parent.name}"
            if not SOURCE_KEY.fullmatch(key):
                continue
            card_data = yaml.safe_load(card.read_text())
            entries[key] = {
                "uri": (card_data.get("uris") or {}).get("all"),
                "card": str(card.relative_to(root)),
                "card_sha256": file_digest(card),
            }
    for key, override in (config.get("sources") or {}).items():
        if not SOURCE_KEY.fullmatch(key):
            raise ValueError(f"Invalid source key {key!r}; expected l<level>/<mask>")
        values = override if isinstance(override, dict) else {"uri": override}
        if "uri" not in values or values.keys() - {"uri", "provenance"}:
            raise ValueError(
                f"Source override {key!r} requires uri and optional provenance"
            )
        entries[key] = entries.get(key, {}) | values
    if not entries:
        raise ValueError("Configure at least one embedding source")
    for key, entry in sorted(entries.items()):
        uri = entry["uri"]
        if isinstance(uri, str) and (match := ENV_URI.fullmatch(uri)):
            uri = os.environ.get(match[1])
        if uri and (not isinstance(uri, str) or "${" in uri):
            raise ValueError(f"Invalid or unsupported URI expression for {key}")
        entry["uri"] = uri or None
        entry["status"] = "available" if uri else "unavailable"
        if not uri:
            entry["reason"] = "uris.all is absent or its environment variable is unset"
    return dict(sorted(entries.items()))


class _InvalidVectorError(ValueError):
    pass


class _Moments:
    """Merge batch means and sums of squared deviations in float64."""

    def __init__(self, width: int) -> None:
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def add(self, batch: np.ndarray) -> None:
        count = len(batch)
        batch_mean = batch.mean(axis=0)
        delta = batch_mean - self.mean
        total = self.count + count
        self.m2 += np.square(batch - batch_mean).sum(axis=0)
        self.m2 += np.square(delta) * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def statistics(self) -> tuple[np.ndarray, np.ndarray]:
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        return self.mean.astype(np.float32), std.astype(np.float32)


def _matrix(values: pa.Array, width: int) -> np.ndarray:
    value_type = values.type
    if not (
        pa.types.is_list(value_type)
        or pa.types.is_large_list(value_type)
        or pa.types.is_fixed_size_list(value_type)
    ):
        raise _InvalidVectorError(
            f"Expected a list-valued embedding column, got {value_type}"
        )
    if values.null_count or not pa.types.is_floating(value_type.value_type):
        raise _InvalidVectorError(
            "Embedding vectors must contain floating point values"
        )
    if not pc.all(pc.equal(pc.list_value_length(values), width)).as_py():
        raise _InvalidVectorError(f"Embedding vector width differs from {width}")
    flat = pc.list_flatten(values)
    if flat.null_count:
        raise _InvalidVectorError("Embedding vectors contain null elements")
    result = np.asarray(flat.to_numpy(), dtype=np.float32)
    if not np.isfinite(result).all():
        raise _InvalidVectorError("Embedding vectors contain nonfinite elements")
    return result.reshape(len(values), width)


def _slide_statistics(
    path: Path,
    slide_id: str,
    width: int,
    batch_size: int,
    epsilon: float,
    sample_count: int,
    sample_salt: bytes,
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
    raw = _Moments(width)
    normalized = _Moments(width)
    samples: list[tuple[int, int, np.ndarray]] = []
    with pq.ParquetFile(path) as parquet:
        if parquet.metadata.num_rows == 0:
            return 0, None, []
        if parquet.schema_arrow.names.count("embedding") != 1:
            raise _InvalidVectorError("Expected exactly one embedding column")
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["embedding"], use_threads=False
        ):
            matrix = _matrix(batch.column(0), width).astype(np.float64)
            if sample_count:
                for row_id, vector in enumerate(matrix, start=raw.count):
                    rank = int.from_bytes(
                        hashlib.blake2b(
                            row_id.to_bytes(8, "little"),
                            key=sample_salt,
                            digest_size=16,
                        ).digest()
                    )
                    # The heap root is the worst retained row.
                    if len(samples) < sample_count:
                        heapq.heappush(
                            samples, (-rank, -row_id, vector.astype(np.float32))
                        )
                    elif (-rank, -row_id) > samples[0][:2]:
                        heapq.heapreplace(
                            samples, (-rank, -row_id, vector.astype(np.float32))
                        )
            raw.add(matrix)
            centered = matrix - matrix.mean(axis=1, keepdims=True)
            normalized.add(
                centered
                / np.sqrt(np.square(centered).mean(axis=1, keepdims=True) + epsilon)
            )
        if parquet.metadata.num_rows != raw.count:
            raise OSError("Embedding row count does not match Parquet metadata")
    mean_raw, std_raw = raw.statistics()
    mean_ln, std_ln = normalized.statistics()
    row = {
        "slide_id": slide_id,
        "mean_raw": mean_raw,
        "std_raw": std_raw,
        "mean_ln": mean_ln,
        "std_ln": std_ln,
        "tile_count": raw.count,
    }
    sample_rows = [
        {"slide_id": slide_id, "row_id": -row_id, "embedding": vector}
        for _, row_id, vector in sorted(samples, reverse=True)
    ]
    return raw.count, row, sample_rows


def _local_dir(uri: str) -> Path | None:
    parsed = urlsplit(uri)
    if not parsed.scheme:
        return Path(uri).expanduser().resolve()
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        return Path(unquote(parsed.path))
    return None


def _list_files(uri: str, directory: Path | None) -> dict[str, dict[str, Any]]:
    files = {}
    if directory is not None:
        if not directory.is_dir():
            raise FileNotFoundError(f"Embedding directory does not exist: {directory}")
        for path in sorted(directory.glob("*.parquet")):
            if path.is_file():
                stat = path.stat()
                files[path.stem] = {
                    "name": path.name,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                }
    else:
        from mlflow.artifacts import list_artifacts

        for file in list_artifacts(artifact_uri=uri):
            path = Path(file.path)
            if file.is_dir or path.suffix != ".parquet":
                continue
            if path.stem in files:
                raise ValueError(f"Duplicate artifact stem {path.stem!r} in {uri}")
            files[path.stem] = {"name": path.name, "bytes": file.file_size}
    return dict(sorted(files.items()))


@contextmanager
def _staged_file(
    uri: str,
    directory: Path | None,
    name: str,
    expected_size: int | None,
    scratch: Path | None,
) -> Iterator[Path]:
    if directory is not None:
        path = directory / name
        if expected_size is not None and path.stat().st_size != expected_size:
            raise OSError(f"Local embedding size changed after inventory: {name}")
        yield path
        return
    from mlflow.artifacts import download_artifacts
    from mlflow.exceptions import MlflowException
    from mlflow.utils.uri import append_to_uri_path

    for attempt in range(3):
        with tempfile.TemporaryDirectory(
            prefix="ridge-slide-", dir=scratch
        ) as temporary:
            try:
                path = Path(
                    download_artifacts(
                        artifact_uri=append_to_uri_path(uri, name), dst_path=temporary
                    )
                )
                if not path.is_file() or (
                    expected_size is not None and path.stat().st_size != expected_size
                ):
                    raise OSError("Staged file does not match the source inventory")
            except (MlflowException, OSError) as exc:
                if attempt == 2:
                    raise OSError(
                        f"MLflow embedding could not be staged after three attempts: {name}"
                    ) from exc
            else:
                yield path
                return
        time.sleep(2**attempt)


def _read_slide(
    uri: str,
    directory: Path | None,
    key: str,
    slide_id: str,
    inputs: Mapping[str, Any],
    settings: Mapping[str, Any],
    scratch: Path | None,
) -> tuple[str, str | None, int | None, dict[str, Any] | None, list[dict[str, Any]]]:
    salt = bytes.fromhex(digest([settings["sample_seed"], key, slide_id]))
    # Staging failures leave the source unfinished. Only a staged Parquet can be
    # classified as unreadable and reconciled as part of a completed source.
    with _staged_file(uri, directory, inputs["name"], inputs["bytes"], scratch) as path:
        try:
            count, row, samples = _slide_statistics(
                path,
                slide_id,
                settings["embedding_dim"],
                settings["batch_size"],
                settings["layernorm_epsilon"],
                settings["sample_tiles"],
                salt,
            )
        except _InvalidVectorError as exc:
            return "invalid-vector", str(exc), None, None, []
        except (OSError, pa.ArrowInvalid) as exc:
            # Record portable reasons without temp paths or remote credentials.
            return "unreadable", type(exc).__name__, None, None, []
    return ("ok" if row else "empty"), None, count, row, samples


def _table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _output_paths(key: str, sampled: bool) -> set[str]:
    files = {f"stats/{split}/{key}.parquet" for split in ("dev", "test")}
    if sampled:
        files.update(f"samples/{split}/{key}.parquet" for split in ("dev", "test"))
    return files


def _valid_outputs(
    directory: Path, files: Mapping[str, str], expected: set[str]
) -> bool:
    return files.keys() == expected and all(
        (directory / name).is_file() and file_digest(directory / name) == checksum
        for name, checksum in files.items()
    )


def _build_source(
    key: str,
    staging: Path,
    source: Mapping[str, Any],
    directory: Path | None,
    files: Mapping[str, Any],
    slides: Mapping[str, Slide],
    settings: Mapping[str, Any],
    scratch: Path | None,
) -> list[dict[str, Any]]:
    vector = pa.list_(pa.float32(), settings["embedding_dim"])
    stats_schema = pa.schema(
        [("slide_id", pa.string()), ("tile_count", pa.int64())]
        + [(name, vector) for name in FEATURES]
    )
    sample_schema = pa.schema(
        [("slide_id", pa.string()), ("row_id", pa.int64()), ("embedding", vector)]
    )
    stats: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    coverage = []
    with ExitStack() as stack:
        samples = {}
        if settings["sample_tiles"]:
            for scope in stats:
                path = staging / f"samples/{scope}/{key}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                samples[scope] = stack.enter_context(
                    pq.ParquetWriter(path, sample_schema)
                )
        candidates = (
            slides.keys() if settings["pilot_slides"] else files.keys() | slides.keys()
        )
        for slide_id in sorted(candidates):
            slide = slides.get(slide_id)
            split = slide.split if slide else None
            count = None
            reason = None
            if slide is None:
                status = "unmapped"
            elif slide_id not in files:
                status = "absent"
            else:
                status, reason, count, row, sampled = _read_slide(
                    source["uri"],
                    directory,
                    key,
                    slide_id,
                    files[slide_id],
                    settings,
                    scratch,
                )
                scope = "test" if split == "test" else "dev"
                if row is not None:
                    stats[scope].append(row)
                if sampled:
                    samples[scope].write_table(
                        pa.Table.from_pylist(sampled, schema=sample_schema)
                    )
            coverage.append(
                {
                    "source": key,
                    "slide_id": slide_id,
                    "split": split,
                    "status": status,
                    "tile_count": count,
                    "reason": reason,
                }
            )
    for scope, rows in stats.items():
        _table(staging / f"stats/{scope}/{key}.parquet", rows, stats_schema)
    return coverage


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = {
        "embedding_dim": config.get("embedding_dim", 2560),
        "batch_size": config.get("batch_size", 1024),
        "layernorm_epsilon": config.get("layernorm_epsilon", 1e-5),
        "sample_tiles": config.get("sample_tiles", 0),
        "sample_seed": config.get("sample_seed", 0),
        "statistics_code_sha256": file_digest(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
    }
    for name in ("embedding_dim", "batch_size", "sample_tiles", "sample_seed"):
        minimum = 1 if name in ("embedding_dim", "batch_size") else 0
        value = settings[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    epsilon = settings["layernorm_epsilon"]
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("layernorm_epsilon must be positive and finite")
    return settings


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    """Check the manifest identity without opening any partition or label data."""
    path = cache_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    payload = {key: value for key, value in manifest.items() if key != "digest"}
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get(
        "digest"
    ) != digest(payload):
        raise ValueError(f"Invalid slide-statistics manifest at {path}")
    return manifest


def _publish_source(
    output: Path,
    staging: Path,
    coverage: list[dict[str, Any]],
    key: str,
    identity: str,
    paths: set[str],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    coverage_path = f".state/coverage/{key}.parquet"
    _table(staging / coverage_path, coverage, COVERAGE_SCHEMA)
    checksums = {}
    for path in sorted(paths | {coverage_path}):
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        (staging / path).replace(target)
        checksums[path] = file_digest(target)
    marker = {
        "identity": identity,
        "inventory": inventory,
        "files": {path: checksums[path] for path in sorted(paths)},
        "coverage": checksums[coverage_path],
    }
    write_json(output / f".state/complete/{key}.json", marker)
    return marker


def _source_result(
    output: Path,
    key: str,
    source: Mapping[str, Any],
    slides: Mapping[str, Slide],
    settings: Mapping[str, Any],
    mapping_sha: str,
    scratch: Path | None,
    reuse: tuple[Path, dict[str, Any], pa.Table] | None,
    published: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    marker_path = output / f".state/complete/{key}.json"
    directory = _local_dir(source["uri"])
    try:
        inventory = _list_files(source["uri"], directory)
    except FileNotFoundError:
        inventory = {}
    if not inventory:
        if marker_path.exists():
            raise ValueError(f"Previously completed source disappeared: {key}")
        return dict(source) | {
            "status": "unavailable",
            "reason": "no accessible slide Parquets in source",
        }, []

    identity = digest([key, source, settings, mapping_sha, inventory])
    paths = _output_paths(key, bool(settings["sample_tiles"]))
    coverage_path = f".state/coverage/{key}.parquet"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        outputs = marker.get("files", {}) | {coverage_path: marker.get("coverage")}
        if (
            marker.get("identity") != identity
            or marker.get("inventory") != inventory
            or not _valid_outputs(output, outputs, paths | {coverage_path})
        ):
            raise ValueError(
                f"Stale or corrupt completion marker for {key}; use a fresh cache"
            )
        coverage = pq.read_table(output / coverage_path).to_pylist()
    else:
        if published:
            raise ValueError(
                f"Source inventory changed or completion marker is missing for {key}"
            )
        with tempfile.TemporaryDirectory(prefix=".source-", dir=output) as temp:
            staging = Path(temp)
            previous = reuse[1]["sources"].get(key, {}) if reuse else {}
            if reuse and previous.get("identity") == identity:
                if not _valid_outputs(reuse[0], previous["files"], paths):
                    raise ValueError(f"Prior cache has corrupt outputs for {key}")
                for path in sorted(paths):
                    target = staging / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Both roots publish immutable content. Link on the same filesystem.
                    try:
                        os.link(reuse[0] / path, target)
                    except OSError:
                        shutil.copyfile(reuse[0] / path, target)
                coverage = (
                    reuse[2].filter(pc.equal(reuse[2]["source"], key)).to_pylist()
                )
            else:
                coverage = _build_source(
                    key,
                    staging,
                    source,
                    directory,
                    inventory,
                    slides,
                    settings,
                    scratch,
                )
            if _list_files(source["uri"], directory) != inventory:
                raise ValueError(f"Embedding source changed during cache build: {key}")
            marker = _publish_source(
                output, staging, coverage, key, identity, paths, inventory
            )
    return dict(source) | {
        "identity": identity,
        "inventory": marker["inventory"],
        "files": marker["files"],
    }, coverage


def _reuse_cache(
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], pa.Table] | None:
    if not (directory := config.get("reuse_cache")):
        return None
    root = Path(directory)
    manifest = load_manifest(root)
    path = root / "coverage.parquet"
    if file_digest(path) != manifest["coverage_sha256"]:
        raise ValueError("Prior cache has a corrupt coverage inventory")
    return root, manifest, pq.read_table(path)


def _prepare_build(
    output: Path, inputs: dict[str, Any], mapping: Path
) -> dict[str, Slide]:
    path = output / ".state/inputs.json"
    slides = load_membership(mapping)
    if path.exists():
        if json.loads(path.read_text()) != inputs:
            raise ValueError("Cache inputs changed; use a fresh cache directory")
    else:
        write_json(path, inputs)
    return slides


def _pilot_diagnostics(
    results: Mapping[str, Any], coverage: list[dict[str, Any]]
) -> dict[str, int]:
    attempted = [
        row
        for row in coverage
        if row["slide_id"] in results[row["source"]].get("inventory", {})
    ]
    sizes = [
        results[row["source"]]["inventory"][row["slide_id"]]["bytes"]
        for row in attempted
    ]
    return {
        "attempted_slide_reads": len(attempted),
        "usable_slide_reads": sum(row["status"] == "ok" for row in attempted),
        "known_input_bytes": sum(size for size in sizes if size is not None),
        "reads_without_known_size": sizes.count(None),
    }


def build_cache(config: Mapping[str, Any], output_dir: Path) -> Path:
    """Stream slide statistics, reconcile inventory, and publish the cache manifest."""
    settings = _settings(config)
    mapping = Path(config["data_mapping"])
    sources = source_inventory(config)
    mapping_sha = file_digest(mapping)
    pilot = config.get("pilot_slides")
    if pilot is not None and (type(pilot) is not int or pilot < 1):
        raise ValueError("pilot_slides must be a positive integer")
    inputs = {
        "schema_version": SCHEMA_VERSION,
        "mapping_sha256": mapping_sha,
        "settings": settings,
        "sources": sources,
        "code_revision": config.get("code_revision"),
        "pilot_slides": pilot,
    }
    scratch = Path(config["scratch_dir"]) if config.get("scratch_dir") else None
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".cache.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is building {output_dir}") from exc
        slides = _prepare_build(output_dir, inputs, mapping)
        if pilot is not None:
            selected = sorted(
                key for key, slide in slides.items() if slide.split != "test"
            )[:pilot]
            if not selected:
                raise ValueError("No development slides available for the cache pilot")
            slides = {key: slides[key] for key in selected}
        reuse = _reuse_cache(config)
        path = output_dir / "manifest.json"
        results, coverage = _build_inventory(
            output_dir,
            sources,
            slides,
            settings | {"pilot_slides": pilot},
            mapping_sha,
            scratch,
            reuse,
            path.exists(),
        )
        if file_digest(mapping) != mapping_sha:
            raise ValueError("Mapping changed during cache build")
        if not any(item["status"] == "available" for item in results.values()):
            raise ValueError("No accessible sources with slide Parquets were found")
        pilot_metadata = {}
        if pilot is not None:
            diagnostics = _pilot_diagnostics(results, coverage)
            if not diagnostics["usable_slide_reads"]:
                raise ValueError(
                    "Cache pilot found no usable development embeddings "
                    f"after {diagnostics['attempted_slide_reads']} slide reads "
                    f"({diagnostics['known_input_bytes']} known input bytes)"
                )
            pilot_metadata["pilot_diagnostics"] = diagnostics
        with tempfile.TemporaryDirectory(prefix=".coverage-", dir=output_dir) as temp:
            coverage_temp = Path(temp) / "coverage.parquet"
            _table(coverage_temp, coverage, COVERAGE_SCHEMA)
            manifest = (
                inputs
                | pilot_metadata
                | {
                    "sources": results,
                    "coverage_sha256": file_digest(coverage_temp),
                }
            )
            if path.exists():
                if load_manifest(output_dir) != manifest | {"digest": digest(manifest)}:
                    raise ValueError(
                        "Published cache inputs changed; use a fresh cache directory"
                    )
                if (
                    file_digest(output_dir / "coverage.parquet")
                    != manifest["coverage_sha256"]
                ):
                    raise ValueError(
                        "Published cache coverage is corrupt; use a fresh cache"
                    )
            else:
                coverage_temp.replace(output_dir / "coverage.parquet")
                write_json(path, manifest | {"digest": digest(manifest)})
            return path


def _build_inventory(
    output: Path,
    sources: Mapping[str, Any],
    slides: Mapping[str, Slide],
    settings: Mapping[str, Any],
    mapping_sha: str,
    scratch: Path | None,
    reuse: tuple[Path, dict[str, Any], pa.Table] | None,
    published: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = {}
    coverage = []
    for key, source in sources.items():
        item = dict(source)
        rows: list[dict[str, Any]] = []
        if item["status"] == "available":
            item, rows = _source_result(
                output,
                key,
                source,
                slides,
                settings,
                mapping_sha,
                scratch,
                reuse,
                published,
            )
        if item["status"] == "unavailable":
            rows = [
                {
                    "source": key,
                    "slide_id": slide_id,
                    "split": slide.split,
                    "status": "source-unavailable",
                    "tile_count": None,
                    "reason": item["reason"],
                }
                for slide_id, slide in sorted(slides.items())
            ]
        results[key] = item
        coverage.extend(rows)
    return results, coverage
