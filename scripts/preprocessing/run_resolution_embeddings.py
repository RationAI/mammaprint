"""Benchmark A40 embedding profiles within an already allocated GPU environment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROFILES = ((1024, 8), (512, 8), (1536, 8), (1024, 4), (1024, 12))


def _benchmark(args: argparse.Namespace) -> int:
    profiles = [(args.batch_size, args.workers)] if args.batch_size else PROFILES
    failed = []
    for batch, workers in profiles:
        name = f"{args.series}-{args.precision}-b{batch}-w{workers}"
        command = [
            sys.executable,
            "-m",
            "preprocessing.resolution_embeddings",
            "--manifest",
            str(args.manifest),
            "--output",
            str(args.output),
            "--encoder-revision",
            args.encoder_revision,
            "--runtime-id",
            args.runtime_id,
            "--pilot",
            name,
            "--batch-size",
            str(batch),
            "--workers",
            str(workers),
            "--prefetch",
            str(args.prefetch),
            "--precision",
            args.precision,
            "--pilot-slides",
            *args.slides,
        ]
        if args.stage_dir is not None:
            command.extend(("--stage-dir", str(args.stage_dir)))
        print(shlex.join(command), flush=True)
        if args.execute and subprocess.run(command, check=False).returncode:
            failed.append(name)
    if failed:
        print(f"Failed profiles: {', '.join(failed)}", file=sys.stderr)
    return int(bool(failed))


def _compare(reference: Path, candidate: Path) -> dict[str, int | list[float]]:
    configs = [
        json.loads((root / "run.json").read_text()) for root in (reference, candidate)
    ]
    keys = (
        "manifest_digest",
        "encoder_revision",
        "normalization",
        "runtime_id",
        "pilot_slides",
    )
    if any(configs[0][key] != configs[1][key] for key in keys):
        raise ValueError("Pilot runs must pin the same tiles, encoder, and runtime")
    if (configs[0]["precision"], configs[1]["precision"]) != ("fp32", "bf16"):
        raise ValueError("Provide the FP32 baseline followed by the BF16 candidate")
    cosine, relative = [], []
    for slide in configs[0]["pilot_slides"]:
        tables = [
            pq.read_table(
                root / "embeddings" / f"{slide}.parquet",
                columns=["x", "y", "embedding"],
            )
            for root in (reference, candidate)
        ]
        if not tables[0].select(["x", "y"]).equals(tables[1].select(["x", "y"])):
            raise ValueError(f"Pilot coordinates differ for {slide}")
        vectors = [
            table["embedding"]
            .combine_chunks()
            .values.to_numpy()
            .reshape(table.num_rows, -1)
            for table in tables
        ]
        norms = [np.linalg.norm(array, axis=1) for array in vectors]
        if (norms[0] <= 0).any() or (norms[1] <= 0).any():
            raise ValueError(f"Zero pilot embedding for {slide}")
        cosine.extend(np.sum(vectors[0] * vectors[1], axis=1) / (norms[0] * norms[1]))
        relative.extend(np.linalg.norm(vectors[0] - vectors[1], axis=1) / norms[0])
    return {
        "slides": len(configs[0]["pilot_slides"]),
        "tiles": len(cosine),
        "cosine_min_p01_median": np.quantile(cosine, [0, 0.01, 0.5]).tolist(),
        "relative_l2_median_p99_max": np.quantile(relative, [0.5, 0.99, 1]).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    pilot = subcommands.add_parser(
        "pilot", help="Render or run bounded benchmark profiles"
    )
    for option in ("manifest", "output"):
        pilot.add_argument(f"--{option}", type=Path, required=True)
    for option in ("series", "encoder-revision", "runtime-id"):
        pilot.add_argument(f"--{option}", required=True)
    pilot.add_argument("--slides", nargs="+", required=True)
    pilot.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    pilot.add_argument("--batch-size", type=int)
    pilot.add_argument("--workers", type=int, default=8)
    pilot.add_argument("--prefetch", type=int, default=2)
    pilot.add_argument("--stage-dir", type=Path)
    pilot.add_argument("--execute", action="store_true")
    comparison = subcommands.add_parser(
        "compare-precision", help="Compare completed pilot vectors"
    )
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "pilot":
        if args.batch_size is not None and args.batch_size < 1:
            parser.error("--batch-size must be positive")
        return _benchmark(args)
    print(json.dumps(_compare(args.reference, args.candidate), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
