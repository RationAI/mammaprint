"""Render a CPU dataset-mask preview Job locally, for a separately approved submission."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence


_REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = _REPOSITORY
CONTEXT = "kubas-cluster"
NAMESPACE = "rationai-jobs-ns"
TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
_TEMPLATE = "2c7a34b549da9a25cdc94178b82d6cfe5de9907b:third_party/rationai-report-worker/pod-template.json"
CHUNK_PREFIX = "MAMMAPRINT_MASK_SOURCE_"
SOURCES = (
    "ml/__init__.py",
    "ml/explainability/dataset_mask_assets.py",
    "ml/explainability/dataset_mask_lineage.py",
    "ml/explainability/dataset_mask_report.py",
    "ml/explainability/tile_removal_report.py",
    "scripts/ml/backfill_dataset_mask_reports.py",
    "scripts/ml/backfill_tile_removal_reports.py",
    "scripts/ml/prepare_dataset_mask_previews.py",
)
_ENV_NAMES = {
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_HTTP_REQUEST_TIMEOUT",
    "HTTPS_PROXY",
}
_REQUIREMENTS = (
    "mlflow==3.10.1",
    "numpy==2.4.4",
    "pyarrow==23.0.1",
    "pyyaml==6.0.3",
    "tifffile==2026.3.3",
    "openslide-bin==4.0.1.2",
    "ratiopath @ git+https://github.com/RationAI/ratiopath.git@8eb007e86388767193e73e89a0743f60254af9ec",
)
_READ_ONLY_MOUNTS = {"data", "public-data", "projects"}
_RESOURCES = {"cpu": "8", "memory": "16Gi", "ephemeral-storage": "20Gi"}


def validate_runs(values: Sequence[str]) -> None:
    if not values or len(values) > 3:
        raise ValueError("Provide from one to three approved report targets.")
    if any(not re.fullmatch(r"[a-f0-9]{32}:[a-f0-9]{64}", run) for run in values):
        raise ValueError(
            "Each --run must contain the run ID and original HTML SHA-256."
        )
    if len({run.split(":", 1)[0] for run in values}) != len(values):
        raise ValueError("Report run IDs must be unique.")


def _pack_sources() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in SOURCES:
            path = ROOT / name
            if (
                not path.is_file()
                or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT.resolve())
            ):
                raise ValueError(
                    f"Expected a regular source file under the repository at {name}."
                )
            source = path.read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(source)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(source))
    return gzip.compress(buffer.getvalue(), mtime=0)


def _pod_template() -> dict[str, Any]:
    raw = subprocess.check_output(["git", "show", _TEMPLATE], cwd=_REPOSITORY)
    template = json.loads(raw)
    if template["spec"]["containers"][0]["image"] != "cerit.io/rationai/base:2.0.6":
        raise ValueError(
            "The expected CPU worker image differs from the pinned template."
        )
    return template


def _worker_command(
    job: str, experiment: str, runs: Sequence[str], digest: str, chunks: list[str]
) -> list[str]:
    decoder = (
        "import base64,hashlib,io,os,tarfile;"
        f"data=base64.b64decode(''.join(os.environ[k] for k in {chunks!r}),validate=True);"
        f"assert hashlib.sha256(data).hexdigest()=={digest!r},'Source archive hash mismatch';"
        "with_archive=tarfile.open(fileobj=io.BytesIO(data),mode='r:gz');"
        "with_archive.extractall('.',filter='data');with_archive.close()"
    )
    command = ["uv", "run", "--no-project", "--python", "3.12"]
    for dependency in _REQUIREMENTS:
        command.extend(["--with", dependency])
    command.extend(
        [
            "python",
            "-m",
            "scripts.ml.prepare_dataset_mask_previews",
            "--experiment-id",
            experiment,
            "--batch",
            job,
            "--stage",
        ]
    )
    for run in runs:
        command.extend(["--run", run])
    return [
        "/bin/bash",
        "-c",
        "\n".join(
            (
                "set +x",
                "set -euo pipefail",
                "umask 077",
                "unset MLFLOW_RUN_ID",
                "export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void",
                "export UV_NO_CACHE=1 UV_TORCH_BACKEND=cpu MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false",
                'export NO_PROXY="mlflow-s3.rationai-mlflow,mlflow.rationai-mlflow,svc.cluster.local,localhost,127.0.0.1"',
                "code_dir=$(mktemp -d /tmp/mammaprint-mask-source.XXXXXX)",
                'cd "$code_dir"',
                shlex.join(["python3", "-c", decoder]),
                "exec " + shlex.join(command),
            )
        ),
    ]


def render(
    job: str, experiment: str, runs: Sequence[str]
) -> tuple[dict[str, Any], str]:
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,48}[a-z0-9])?", job):
        raise ValueError(
            "Expected a Kubernetes DNS-label job name of at most 50 characters."
        )
    if not re.fullmatch(r"0|[1-9][0-9]*", experiment):
        raise ValueError("Expected a numeric MLflow experiment ID.")
    validate_runs(runs)
    blob = _pack_sources()
    digest = hashlib.sha256(blob).hexdigest()
    encoded = base64.b64encode(blob).decode()
    env_chunks = [
        {
            "name": f"{CHUNK_PREFIX}{offset // 32000:03d}",
            "value": encoded[offset : offset + 32000],
        }
        for offset in range(0, len(encoded), 32000)
    ]
    pod = _pod_template()
    pod["metadata"]["labels"]["app"] = job
    spec = pod["spec"]
    spec["automountServiceAccountToken"] = False
    container = spec["containers"][0]
    container["name"] = "preview"
    container["env"] = (
        [{"name": "MLFLOW_TRACKING_URI", "value": TRACKING_URI}]
        + [entry for entry in container["env"] if entry["name"] in _ENV_NAMES]
        + env_chunks
    )
    container["command"] = _worker_command(
        job, experiment, runs, digest, [entry["name"] for entry in env_chunks]
    )
    container["resources"] = {
        "requests": dict(_RESOURCES),
        "limits": dict(_RESOURCES),
    }
    for mount in container["volumeMounts"]:
        if mount["name"] in _READ_ONLY_MOUNTS:
            mount["readOnly"] = True
    for volume in spec["volumes"]:
        if volume["name"] in _READ_ONLY_MOUNTS:
            volume["persistentVolumeClaim"]["readOnly"] = True
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job, "namespace": NAMESPACE},
        "spec": {
            "parallelism": 1,
            "completions": 1,
            "backoffLimit": 0,
            "activeDeadlineSeconds": 10800,
            "ttlSecondsAfterFinished": 604800,
            "template": pod,
        },
    }, digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, choices=[CONTEXT])
    parser.add_argument("--namespace", required=True, choices=[NAMESPACE])
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        job, payload_digest = render(args.job_name, args.experiment_id, args.run)
        manifest = (json.dumps(job, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(
            os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
        ) as file:
            file.write(manifest)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "context": args.context,
                "namespace": args.namespace,
                "job_name": args.job_name,
                "experiment_id": args.experiment_id,
                "tracking_uri": TRACKING_URI,
                "run_ids": [run.split(":", 1)[0] for run in args.run],
                "payload_sha256": payload_digest,
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
