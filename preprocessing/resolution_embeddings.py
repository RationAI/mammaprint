"""Resumable Virchow2 worker for a frozen resolution-comparison manifest."""

from __future__ import annotations

import argparse
import bisect
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from preprocessing.embedding_artifacts import EmbeddingArtifacts


if TYPE_CHECKING:
    from collections.abc import Iterator

    from ratiopath.openslide import OpenSlide


FEATURES = 2560
type TileRead = tuple[NDArray[np.uint8] | None, int, int, str | None, float]
type Batch = tuple[torch.Tensor | None, list[tuple[int, int, str | None]], float]


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock(path: Path, *, blocking: bool = False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            mode = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
            fcntl.flock(stream, mode)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}") from exc
        yield


def _safe_id(value: str) -> str:
    if value in ("", ".", "..") or Path(value).name != value:
        raise ValueError(f"Invalid slide or profile ID {value!r}")
    return value


def _manifest(path: Path) -> dict[str, Any]:
    manifest = load_selection_manifest(path)
    ids = [_safe_id(slide["slide_id"]) for slide in manifest["slides"]]
    if manifest["level"] != 1 or len(ids) != len(set(ids)):
        raise ValueError("Expected an L1 manifest with unique slide IDs")
    return manifest


def load_selection_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    contents = {key: value for key, value in manifest.items() if key != "digest"}
    digest = hashlib.sha256(
        json.dumps(contents, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if manifest.get("digest") != digest:
        raise ValueError(f"Selection manifest digest mismatch for {path}")
    return manifest


def _selection(root: Path, slide: dict[str, Any]) -> pa.Table:
    path = root / "tiles" / f"{slide['slide_id']}.parquet"
    table = pq.read_table(path)
    if table.num_rows != slide["tile_count"]:
        raise ValueError(f"Selection row count changed for {slide['slide_id']}")
    return table


class ResolutionTiles(Dataset[TileRead]):
    """Read spatially ordered tiles with one reusable WSI handle per worker."""

    def __init__(
        self, selection: Path, slides: list[dict[str, Any]], staged: dict[str, str]
    ):
        self.selection = selection
        self.slides = slides
        self.staged = staged
        self.ends = np.cumsum([slide["tile_count"] for slide in slides]).tolist()
        self.current = -1
        self.reader: OpenSlide | None = None
        self.error: str | None = None
        self.xy: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def _open(self, index: int) -> None:
        from ratiopath.openslide import OpenSlide

        from preprocessing.embedding_selection import selection_digest

        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.current, self.error = index, None
        slide = self.slides[index]
        try:
            table = _selection(self.selection, slide)
            if selection_digest(table) != slide["selection_digest"]:
                raise ValueError(f"Selection digest changed for {slide['slide_id']}")
            self.xy = (table["x"].to_numpy(), table["y"].to_numpy())
            self.reader = OpenSlide(self.staged.get(slide["slide_id"], slide["path"]))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def __getitem__(self, index: int) -> TileRead:
        start = time.perf_counter()
        group = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[group - 1] if group else 0)
        if self.current != group:
            self._open(group)
        slide = self.slides[group]
        image = None
        if self.error is None:
            assert self.reader is not None and self.xy is not None
            try:
                image = self.reader.read_tile(
                    int(self.xy[0][offset]),
                    int(self.xy[1][offset]),
                    slide["tile_extent_x"],
                    slide["tile_extent_y"],
                    slide["level"],
                )
                if image.shape != (slide["tile_extent_y"], slide["tile_extent_x"], 3):
                    raise ValueError(f"Unexpected image shape {image.shape}")
            except Exception as exc:
                image = None
                self.error = f"{type(exc).__name__}: {exc}"
        return image, group, offset, self.error, time.perf_counter() - start


def _collate(samples: list[TileRead]) -> Batch:
    images = [image for image, *_ in samples if image is not None]
    keys = [(group, offset, error) for _, group, offset, error, _ in samples]
    return (
        torch.from_numpy(np.stack(images)) if images else None,
        keys,
        sum(seconds for *_, seconds in samples),
    )


def _normalize(
    images: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return images.float().sub_(mean).mul_(scale).permute(0, 3, 1, 2)


def _paths(output: Path, slide: dict[str, Any]) -> tuple[Path, Path]:
    stem = slide["slide_id"]
    return (
        output / "embeddings" / f"{stem}.parquet",
        output / "complete" / f"{stem}.json",
    )


def is_complete(output: Path, slide: dict[str, Any]) -> bool:
    parquet, record = _paths(output, slide)
    if not (parquet.is_file() and record.is_file()):
        return False
    try:
        completion = json.loads(record.read_text())
        metadata = pq.read_metadata(parquet)
        return (
            completion["selection_digest"] == slide["selection_digest"]
            and metadata.num_rows == slide["tile_count"] == completion["tiles"]
            and pq.read_schema(parquet).field("embedding").type
            == pa.list_(pa.float32(), FEATURES)
            and completion["sha256"] == file_digest(parquet)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _publish_slide(
    selection: Path,
    output: Path,
    slide: dict[str, Any],
    features: np.ndarray,
    artifacts: EmbeddingArtifacts | None = None,
) -> tuple[int, float]:
    from preprocessing.embedding_selection import selection_digest

    start = time.perf_counter()
    if (
        features.shape != (slide["tile_count"], FEATURES)
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"Invalid Virchow2 feature array for {slide['slide_id']}")
    table = _selection(selection, slide)
    if selection_digest(table) != slide["selection_digest"]:
        raise ValueError(f"Selection digest changed for {slide['slide_id']}")
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(features.ravel()), FEATURES)
    parquet, record = _paths(output, slide)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = parquet.with_name(f".{parquet.name}.{uuid4().hex}")
    try:
        pq.write_table(table.append_column("embedding", vectors), temporary)
        digest = file_digest(temporary)
        if pq.read_metadata(temporary).num_rows != table.num_rows:
            raise ValueError(f"Invalid Parquet footer for {slide['slide_id']}")
        os.replace(temporary, parquet)
        write_json(
            record,
            {
                "selection_digest": slide["selection_digest"],
                "tiles": table.num_rows,
                "sha256": digest,
            },
        )
        if artifacts is not None:
            artifacts.upload(parquet, record)
    finally:
        temporary.unlink(missing_ok=True)
    return table.num_rows, time.perf_counter() - start


def _read_stage(
    slides: list[dict[str, Any]], root: Path | None
) -> tuple[dict[str, str], float]:
    start = time.perf_counter()
    if root is None:
        return {}, 0.0
    staged = {}
    for slide in slides:
        source = Path(slide["path"])
        if source.suffix.lower() not in {".svs", ".tif", ".tiff", ".ndpi"}:
            raise ValueError(
                f"Staging only supports single-file formats, got {source.suffix}"
            )
        target = root / f"{slide['slide_id']}{source.suffix}"
        shutil.copyfile(source, target)
        staged[slide["slide_id"]] = str(target)
    return staged, time.perf_counter() - start


def _environment(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {
        "encoder_revision": args.encoder_revision,
        "runtime_id": args.runtime_id,
        "precision": args.precision,
        "normalization": "imagenet-rgb-fp32-v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
    }


def _infer(
    args: argparse.Namespace,
    selection: Path,
    output: Path,
    slides: list[dict[str, Any]],
    summary: dict[str, Any],
    device: torch.device,
    model: torch.nn.Module,
    staged: dict[str, str],
    artifacts: EmbeddingArtifacts | None = None,
) -> None:
    dataset = ResolutionTiles(selection, slides, staged)
    multiprocessing = (
        {
            "prefetch_factor": args.prefetch,
            "persistent_workers": True,
            "multiprocessing_context": "spawn",
            "timeout": args.loader_timeout,
        }
        if args.workers
        else {}
    )
    cuda = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=cuda,
        collate_fn=_collate,
        **multiprocessing,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device) * 255
    scale = (torch.tensor([0.229, 0.224, 0.225], device=device) * 255).reciprocal()
    events = (
        tuple(torch.cuda.Event(enable_timing=True) for _ in range(4)) if cuda else ()
    )
    summary["batches"] = 0

    write_queue: deque[Future[tuple[int, float]]] = deque()

    def finish_write() -> None:
        start = time.perf_counter()
        tiles, seconds = write_queue.popleft().result()
        summary["committed_tiles"] += tiles
        summary["write_worker_seconds"] += seconds
        summary["writer_wait_seconds"] += time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=1) as writer, torch.inference_mode():
        iterator = iter(loader)
        # Input order is fixed, so at most one partially assembled slide spans batches.
        group = -1
        features = np.empty((0, FEATURES), dtype=np.float32)
        error = None
        while True:
            start = time.perf_counter()
            try:
                images, keys, read_seconds = next(iterator)
            except StopIteration:
                break
            summary["loader_wait_seconds"] += time.perf_counter() - start
            summary["read_worker_seconds"] += read_seconds
            summary["batches"] += 1
            if images is not None:
                if events:
                    transfers, h2d_end, inference_end, output_end = events
                    transfers.record()
                images = images.to(device, non_blocking=cuda)
                if events:
                    h2d_end.record()
                compute_start = time.perf_counter()
                precision = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.precision == "bf16"
                    else nullcontext()
                )
                with precision:
                    encoded = model(_normalize(images, mean, scale))
                if events:
                    inference_end.record()
                    vectors_host = torch.empty(
                        encoded.shape, dtype=torch.float32, pin_memory=True
                    )
                    vectors_host.copy_(encoded.float(), non_blocking=True)
                    output_end.record()
                    output_end.synchronize()
                else:
                    vectors_host = encoded.float()
                vectors = vectors_host.numpy()
                if events:
                    summary["h2d_seconds"] += transfers.elapsed_time(h2d_end) / 1000
                    summary["gpu_seconds"] += h2d_end.elapsed_time(inference_end) / 1000
                    summary["d2h_seconds"] += (
                        inference_end.elapsed_time(output_end) / 1000
                    )
                else:
                    summary["gpu_seconds"] += time.perf_counter() - compute_start
            position = 0
            for index, offset, problem in keys:
                if group != index:
                    group = index
                    error = None
                    features = np.empty(
                        (slides[group]["tile_count"], FEATURES), dtype=np.float32
                    )
                if problem is None:
                    features[offset] = vectors[position]
                    position += 1
                elif error is None:
                    error = problem
                if offset + 1 == len(features):
                    slide = slides[group]
                    if error is not None:
                        summary["failures"][slide["slide_id"]] = error
                    else:
                        if len(write_queue) >= 2:
                            finish_write()
                        write_queue.append(
                            writer.submit(
                                _publish_slide,
                                selection,
                                output,
                                slide,
                                features,
                                artifacts,
                            )
                        )
        while write_queue:
            finish_write()


def _run(args: argparse.Namespace) -> int:
    selection = args.manifest.resolve().parent
    manifest = _manifest(args.manifest)
    if args.mlflow_run_id and (args.pilot is not None or manifest["shards"] != 1):
        raise ValueError("Direct MLflow storage requires a single production shard")
    if args.pilot is not None:
        requested = set(args.pilot_slides or [])
        available = {
            slide["slide_id"]
            for slide in manifest["slides"]
            if slide["split"] != "test" and slide["tile_count"] > 0
        }
        if not requested or requested - available:
            raise ValueError(
                "Provide nonempty train/validation --pilot-slides from the manifest"
            )
        items = [
            slide for slide in manifest["slides"] if slide["slide_id"] in requested
        ]
        output = args.output / "pilot" / _safe_id(args.pilot)
        job = "pilot"
    else:
        if args.shard is None or args.shard < 0:
            raise ValueError("Supply a nonnegative --shard for a production worker")
        items = [slide for slide in manifest["slides"] if slide["shard"] == args.shard]
        if not items:
            raise ValueError(f"Shard {args.shard} has no manifest entries")
        output = args.output / "production"
        job = f"shard-{args.shard}"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "job": job,
                    "output": str(output),
                    "slides": len(items),
                    "tiles": sum(slide["tile_count"] for slide in items),
                    "manifest_digest": manifest["digest"],
                    "batch_size": args.batch_size,
                    "workers": args.workers,
                    "prefetch": args.prefetch,
                    "loader_timeout": args.loader_timeout,
                    "precision": args.precision,
                },
                indent=2,
            )
        )
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this allocation")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 requires a CUDA device with BF16 support")
        if args.pilot is None and not args.allow_bf16:
            raise ValueError(
                "Use --allow-bf16 after validating pilot precision agreement"
            )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    begin = time.perf_counter()
    with lock(output / "locks" / f"{job}.lock"):
        config = {**_environment(args, device), "manifest_digest": manifest["digest"]}
        if args.pilot is not None:
            config.update(
                pilot_slides=sorted(requested),
                batch_size=args.batch_size,
                workers=args.workers,
                prefetch=args.prefetch,
                stage=bool(args.stage_dir),
            )
        state_path = output / "run.json"
        with lock(output / "locks" / "config.lock", blocking=True):
            if state_path.exists() and json.loads(state_path.read_text()) != config:
                raise ValueError(
                    f"Output directory is pinned to a different workload: {output}"
                )
            if not state_path.exists():
                write_json(state_path, config)
        validation_start = time.perf_counter()
        artifacts = None
        if args.mlflow_run_id:
            from mlflow import MlflowClient

            artifacts = EmbeddingArtifacts(
                MlflowClient(), args.mlflow_run_id, output / ".mlflow"
            )
            artifacts.prepare(config)
        pending = [
            slide
            for slide in items
            if slide["tile_count"]
            and not (
                artifacts.complete(slide) if artifacts else is_complete(output, slide)
            )
        ]
        summary = {
            "job": job,
            "manifest_digest": manifest["digest"],
            "batch_size": args.batch_size,
            "workers": args.workers,
            "prefetch": args.prefetch,
            "loader_timeout": args.loader_timeout,
            "precision": args.precision,
            "device": config["device"],
            "slides": len(items),
            "attempted_tiles": sum(slide["tile_count"] for slide in pending),
            "committed_tiles": 0,
            "resume_check_seconds": time.perf_counter() - validation_start,
            "loader_wait_seconds": 0.0,
            "read_worker_seconds": 0.0,
            "h2d_seconds": 0.0,
            "gpu_seconds": 0.0,
            "d2h_seconds": 0.0,
            "stage_seconds": 0.0,
            "write_worker_seconds": 0.0,
            "writer_wait_seconds": 0.0,
            "encoder_load_seconds": 0.0,
            "failures": {},
            "empty": [slide["slide_id"] for slide in items if not slide["tile_count"]],
        }
        try:
            if pending:
                from preprocessing.embeddings import Virchow2

                encoder_start = time.perf_counter()
                model = Virchow2(revision=args.encoder_revision).eval().to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                summary["encoder_load_seconds"] = time.perf_counter() - encoder_start
                stage = (
                    tempfile.TemporaryDirectory(dir=args.stage_dir)
                    if args.stage_dir is not None
                    else nullcontext(None)
                )
                with stage as directory:
                    staged, summary["stage_seconds"] = _read_stage(
                        pending, Path(directory) if directory is not None else None
                    )
                    _infer(
                        args,
                        selection,
                        output,
                        pending,
                        summary,
                        device,
                        model,
                        staged,
                        artifacts,
                    )
        except Exception as exc:
            summary["fatal"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - begin
            summary["wall_seconds"] = elapsed
            summary["tiles_per_second"] = summary["committed_tiles"] / max(
                elapsed, 1e-6
            )
            processing = elapsed - summary["encoder_load_seconds"]
            summary["tiles_per_second_excluding_model_load"] = summary[
                "committed_tiles"
            ] / max(processing, 1e-6)
            if device.type == "cuda":
                summary["peak_cuda_bytes"] = torch.cuda.max_memory_reserved(device)
            report = (
                output / "attempts" / job / f"{time.time_ns()}-{uuid4().hex[:8]}.json"
            )
            write_json(report, summary)
            print(json.dumps({"report": str(report), **summary}, indent=2))
            if artifacts is not None:
                artifacts.client.log_artifact(
                    artifacts.run_id, str(report), artifact_path="attempts"
                )
                inventory_ok = artifacts.report(output / "l1_inventory.json", items)
            else:
                inventory_ok = True
        return int(bool(summary["failures"]) or not inventory_ok)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot")
    mode.add_argument("--shard", type=int)
    parser.add_argument("--pilot-slides", nargs="+")
    parser.add_argument(
        "--encoder-revision", required=True, help="Virchow2 HF commit SHA"
    )
    parser.add_argument(
        "--runtime-id", required=True, help="Pinned image and software identifier"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument(
        "--loader-timeout",
        type=int,
        default=300,
        help="Seconds to wait for a worker batch",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--allow-bf16", action="store_true")
    parser.add_argument(
        "--stage-dir", type=Path, help="Optional pilot-only local WSI staging"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mlflow-run-id", help="Stream completed slides to this MLflow run"
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not re.fullmatch(r"[a-fA-F0-9]{40}", args.encoder_revision):
        parser.error(
            "--encoder-revision must be a 40-character Hugging Face commit SHA"
        )
    if args.batch_size < 1 or args.workers < 0 or args.prefetch < 1:
        parser.error(
            "Provide a positive batch size and prefetch and nonnegative worker count"
        )
    if args.loader_timeout < 1:
        parser.error("--loader-timeout must be positive")
    if args.stage_dir is not None and args.pilot is None:
        parser.error(
            "--stage-dir is pilot-only; adopt staging only after checking scratch capacity"
        )
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
