"""Render an offline Kubernetes Job for one L2 embedding view on an A40."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import PurePosixPath
from typing import Any

from kube_jobs import storage
from kube_jobs.constants import (
    CLUSTER,
    HTTPS_PROXY,
    JOB_NAMESPACE,
    NO_PROXY,
    TTL_SECONDS_AFTER_FINISHED,
)
from kube_jobs.resolve_gpu import resolve_gpu
from kube_jobs.resolve_storage import resolve_storage
from kubernetes import client


IMAGE = "cerit.io/rationai/base:2.0.6"
TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
PROJECTS = PurePosixPath("/mnt/projects")
PROJECT = PROJECTS / "mammaprint"
RESAMPLED_WSIS = PROJECTS / "quality_control/resampled_mammaprint"
DATA = PurePosixPath("/mnt/data")
TELEMETRY = ".telemetry"
MIB = 1024**2
GIB = 1024**3
SCRATCH_MARGIN = 8 * GIB

BOOTSTRAP = r"""set +x
set -euo pipefail
source_commit=$1
telemetry_root=$2
sample_seconds=$3
minimum_shm_bytes=$4
shift 4
telemetry_dir="$telemetry_root/${POD_UID:?Missing pod UID}"
mkdir -p "$telemetry_root"
mkdir "$telemetry_dir"

worker_pid=
sampler_pid=
on_exit() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$sampler_pid" ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sample || printf 'Final telemetry sample failed\n' >&2
    fi
    printf '%s\n' "$status" > "$telemetry_dir/exit-status" || true
    exit "$status"
}
on_signal() {
    local signal=$1 status=$2
    trap '' INT TERM
    if [[ -n "$worker_pid" ]]; then
        kill -s "$signal" "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap on_exit EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT

# Find the container's own memory cgroup in either cgroup version. The PID
# check avoids accidentally using host memory counters in a shared namespace.
memory_current=
memory_peak=
while IFS=: read -r hierarchy controllers cgroup; do
    if [[ "$hierarchy" == 0 && -z "$controllers" ]]; then
        mount=/sys/fs/cgroup
        current=memory.current
        peak=memory.peak
    elif [[ ",$controllers," == *,memory,* ]]; then
        mount=/sys/fs/cgroup/memory
        current=memory.usage_in_bytes
        peak=memory.max_usage_in_bytes
    else
        continue
    fi
    for directory in "$mount$cgroup" "$mount"; do
        if [[ -r "$directory/$current" ]] && grep -qx "$$" "$directory/cgroup.procs"; then
            memory_current="$directory/$current"
            memory_peak="$directory/$peak"
            break 2
        fi
    done
done < /proc/self/cgroup
if [[ -z "$memory_current" ]]; then
    printf 'Could not identify the container memory cgroup\n' >&2
    exit 1
fi

visible_gpus=$(timeout 10s nvidia-smi --query-gpu=uuid,name --format=csv,noheader)
if [[ "$visible_gpus" == *$'\n'* || "$visible_gpus" != GPU-*,*A40* ]]; then
    printf 'Expected exactly one visible A40, got %s\n' "$visible_gpus" >&2
    exit 1
fi
gpu_uuid=${visible_gpus%%,*}
shm_stats() {
    df -Pk /dev/shm | awk 'NR == 2 { printf "%.0f\t%.0f\n", $3*1024, $2*1024 }'
}
read -r _ shm_total < <(shm_stats)
if (( shm_total < minimum_shm_bytes )); then
    printf 'Mounted /dev/shm is smaller than the required %s bytes\n' "$minimum_shm_bytes" >&2
    exit 1
fi
printf 'unix_seconds\tcgroup_memory_bytes\tcgroup_peak_bytes\tshm_used_bytes\tshm_total_bytes\n' \
    > "$telemetry_dir/resources.tsv"
printf 'unix_seconds,gpu_uuid,gpu_util_percent,gpu_used_mib,gpu_total_mib\n' > "$telemetry_dir/gpu.csv"
sample() {
    local timestamp ram peak="" shm gpu
    timestamp=$(date -u +%s) || return
    ram=$(cat "$memory_current") || return
    if [[ -r "$memory_peak" ]]; then
        peak=$(cat "$memory_peak") || return
    fi
    shm=$(shm_stats) || return
    printf '%s\t%s\t%s\t%s\n' "$timestamp" "$ram" "$peak" "$shm" >> "$telemetry_dir/resources.tsv" || return
    gpu=$(timeout 10s nvidia-smi --id="$gpu_uuid" --query-gpu=uuid,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits) || return
    printf '%s,%s\n' "$timestamp" "$gpu" >> "$telemetry_dir/gpu.csv"
}
sample
(
    while sleep "$sample_seconds"; do
        sample || printf 'Telemetry sample failed\n' >&2
    done
) &
sampler_pid=$!

export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=credential.https://gitlab.ics.muni.cz.helper
export GIT_CONFIG_VALUE_0='!f() { if [ "$1" = get ]; then printf "%s\n" "username=$VCS_USER" "password=$VCS_PASSWORD"; fi; }; f'
git clone --no-checkout https://github.com/RationAI/mammaprint.git /scratch/mammaprint
cd /scratch/mammaprint
git checkout --detach "$source_commit"
uv sync --frozen
unset VCS_USER VCS_PASSWORD GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
"$@" &
worker_pid=$!
wait "$worker_pid"
"""


def _hex_id(value: str, length: int) -> str:
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise argparse.ArgumentTypeError(f"Expected {length} lowercase hex digits")
    return value


def _dns_label(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value):
        raise argparse.ArgumentTypeError("Expected a Kubernetes DNS label")
    return value


def _identifier(value: str) -> str:
    if (
        value in ("", ".", "..")
        or "/" in value
        or value.startswith("-")
        or not value.isprintable()
    ):
        raise argparse.ArgumentTypeError("Expected a safe pilot name or slide ID")
    return value


def _absolute_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.root != "/"
        or str(path) != value
        or ".." in path.parts
        or not value.isprintable()
    ):
        raise argparse.ArgumentTypeError("Expected a normalized absolute pod path")
    return path


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def _size_bytes(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)(Gi|Mi)", value)
    if not match:
        raise ValueError(
            f"Expected a positive whole Mi or Gi resource size, got {value}"
        )
    return int(match[1]) * (GIB if match[2] == "Gi" else MIB)


def _validate(args: argparse.Namespace) -> tuple[dict[str, str], int]:
    if PROJECT not in args.manifest.parents and DATA not in args.manifest.parents:
        raise ValueError(
            "--manifest must be below /mnt/projects/mammaprint or /mnt/data"
        )
    if PROJECT not in args.output_root.parents:
        raise ValueError("--output-root must be below /mnt/projects/mammaprint")
    if args.output_root not in args.output.parents:
        raise ValueError("--output must be below --output-root")
    if args.output.relative_to(args.output_root).parts[0] == TELEMETRY:
        raise ValueError(f"--output cannot use the reserved {TELEMETRY} directory")
    selection = args.manifest.parent
    if selection.is_relative_to(args.output_root) or args.output_root.is_relative_to(
        selection
    ):
        raise ValueError(
            "The manifest directory and writable --output-root must be disjoint"
        )
    if bool(args.pilot) != bool(args.pilot_slides):
        raise ValueError(
            "--pilot and at least one --pilot-slides must be supplied together"
        )
    if len(args.pilot_slides) != len(set(args.pilot_slides)):
        raise ValueError("Pilot slide IDs must be unique")
    if args.pilot and args.mlflow_run_id:
        raise ValueError("--mlflow-run-id requires a production job without --pilot")
    if args.seed < 0 or args.workers < 0:
        raise ValueError("Expected nonnegative --seed and --workers")

    shm = _size_bytes(args.shm)
    memory = _size_bytes(args.memory)
    scratch = _size_bytes(args.scratch_size)
    ephemeral = args.ephemeral_storage
    if ephemeral is None:
        ephemeral = (
            f"{(scratch + SCRATCH_MARGIN) // GIB}Gi"
            if scratch % GIB == 0
            else f"{(scratch + SCRATCH_MARGIN) // MIB}Mi"
        )
    if _size_bytes(ephemeral) < scratch + SCRATCH_MARGIN:
        raise ValueError(
            "--ephemeral-storage must cover scratch plus 8Gi for logs and the container"
        )

    batch_bytes = args.batch_size * 3 * 224**2
    # Reserve two active batches, 25% above the queue payload, and 1Gi for IPC.
    minimum_shm = 0
    if args.workers:
        buffered_batches = args.workers * args.prefetch + 2
        minimum_shm = (buffered_batches * batch_bytes * 5 + 3) // 4 + GIB
    if shm < minimum_shm:
        raise ValueError(
            f"--shm must be at least {minimum_shm / GIB:.2f} GiB for this profile"
        )
    # The tmpfs counts against the memory cgroup. Budget two pinned batches,
    # 512Mi per reader, and 8Gi for parent/writer processes and file cache.
    minimum_memory = shm + 2 * batch_bytes + args.workers * 512 * MIB + 8 * GIB
    if memory < minimum_memory:
        raise ValueError(
            f"--memory must be at least {minimum_memory / GIB:.2f} GiB for this profile"
        )
    return {
        "cpu": str(args.cpu),
        "memory": args.memory,
        "ephemeral-storage": ephemeral,
    }, minimum_shm


def _worker(args: argparse.Namespace) -> list[str]:
    runtime = f"{args.image}@git:{args.source_commit}"
    if args.runtime_id:
        runtime = f"{runtime}/{args.runtime_id}"
    worker = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "preprocessing.l2_augmentation",
        "embed",
        "--manifest",
        str(args.manifest),
        "--output",
        str(args.output),
        "--view",
        args.view,
        "--seed",
        str(args.seed),
        "--encoder-revision",
        args.encoder_revision,
        "--runtime-id",
        runtime,
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--prefetch",
        str(args.prefetch),
    ]
    if args.pilot:
        worker.extend(("--pilot", args.pilot, "--pilot-slides", *args.pilot_slides))
    if args.mlflow_run_id:
        worker.extend(("--mlflow-run-id", args.mlflow_run_id))
    return worker


def _environment() -> list[dict[str, Any]]:
    values = {
        "MLFLOW_TRACKING_URI": TRACKING_URI,
        "MLFLOW_TRACKING_USERNAME": "kissmi",
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "600",
        "HTTPS_PROXY": HTTPS_PROXY,
        "NO_PROXY": f"{NO_PROXY},mlflow-s3.rationai-mlflow",
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": "/scratch",
        "HF_HOME": "/scratch/huggingface",
        "UV_CACHE_DIR": "/scratch/uv-cache",
    }
    return [
        *({"name": name, "value": value} for name, value in values.items()),
        *(
            {"name": name, "valueFrom": {"secretKeyRef": {"name": secret, "key": key}}}
            for name, secret, key in (
                ("HF_TOKEN", "huggingface-secret", "token"),
                ("VCS_USER", "gitlab-deploy-token", "username"),
                ("VCS_PASSWORD", "gitlab-deploy-token", "password"),
            )
        ),
        {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
    ]


def render_job(args: argparse.Namespace) -> dict[str, Any]:
    limits, minimum_shm = _validate(args)
    gpu = resolve_gpu("A40")
    resources = {**limits, **gpu.limits}
    mounts, volumes = resolve_storage([storage.secure.DATA, storage.secure.PROJECTS])
    with client.ApiClient() as api:
        volume_mounts = api.sanitize_for_serialization(mounts)
        volume_specs = api.sanitize_for_serialization(volumes)
    for mount in volume_mounts:
        mount["readOnly"] = True
        if mount["name"] == "projects":
            mount.update(
                mountPath=str(PROJECT), subPath=str(PROJECT.relative_to(PROJECTS))
            )
    for volume in volume_specs:
        if volume["name"] in ("data", "public-data"):
            volume["persistentVolumeClaim"]["readOnly"] = True
    volume_mounts.extend(
        [
            {
                "name": "projects",
                "mountPath": str(RESAMPLED_WSIS),
                "subPath": str(RESAMPLED_WSIS.relative_to(PROJECTS)),
                "readOnly": True,
            },
            {
                "name": "projects",
                "mountPath": str(args.output_root),
                "subPath": str(args.output_root.relative_to(PROJECTS)),
            },
            {"name": "shm", "mountPath": "/dev/shm"},
            {"name": "scratch", "mountPath": "/scratch"},
        ]
    )
    volume_specs.extend(
        [
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": args.shm}},
            {"name": "scratch", "emptyDir": {"sizeLimit": args.scratch_size}},
        ]
    )
    command = [
        BOOTSTRAP,
        args.job_name,
        args.source_commit,
        str(args.output_root / TELEMETRY / args.job_name),
        str(args.sample_seconds),
        str(minimum_shm),
        *_worker(args),
    ]
    # Kubelet performs $(NAME) expansion even in argv that Bash treats literally.
    command = [value.replace("$", "$$") for value in command]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": args.job_name,
            "namespace": args.namespace,
            "labels": {"created_by": "kissmi"},
            "annotations": {
                "mammaprint/source-commit": args.source_commit,
                "mammaprint/cluster-context": args.context,
            },
        },
        "spec": {
            "parallelism": 1,
            "completions": 1,
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": TTL_SECONDS_AFTER_FINISHED,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "nodeSelector": gpu.node_selector,
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "volumes": volume_specs,
                    "containers": [
                        {
                            "name": "l2-embedding",
                            "image": args.image,
                            "imagePullPolicy": "Always",
                            "command": ["/bin/bash", "-c"],
                            "args": command,
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {"requests": resources, "limits": resources},
                            "volumeMounts": volume_mounts,
                            "env": _environment(),
                        }
                    ],
                },
            },
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--context", default=CLUSTER)
    parser.add_argument("--namespace", type=_dns_label, default=JOB_NAMESPACE)
    parser.add_argument("--job-name", type=_dns_label, required=True)
    parser.add_argument(
        "--source-commit",
        type=lambda sha: _hex_id(sha, 40),
        required=True,
        help="Full SHA of the published MammaPrint commit containing the worker",
    )
    parser.add_argument("--manifest", type=_absolute_path, required=True)
    parser.add_argument(
        "--output-root",
        type=_absolute_path,
        required=True,
        help="Precreated writable project directory for results and per-pod telemetry",
    )
    parser.add_argument("--output", type=_absolute_path, required=True)
    parser.add_argument(
        "--view",
        choices=("clean", "stain", "rotation", "stain-rotation"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--encoder-revision", type=lambda sha: _hex_id(sha, 40), required=True
    )
    parser.add_argument("--runtime-id")
    parser.add_argument(
        "--image",
        default=IMAGE,
        help="RationAI base image, preferably pinned by its registry SHA-256 digest",
    )
    parser.add_argument("--pilot", type=_identifier)
    parser.add_argument(
        "--pilot-slides", action="extend", nargs="+", type=_identifier, default=[]
    )
    parser.add_argument("--mlflow-run-id", type=lambda run: _hex_id(run, 32))
    parser.add_argument("--batch-size", type=_positive_int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=_positive_int, default=2)
    parser.add_argument("--cpu", type=_positive_int, default=24)
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--shm", default="12Gi")
    parser.add_argument("--scratch-size", default="32Gi")
    parser.add_argument(
        "--ephemeral-storage", help="Defaults to --scratch-size plus 8Gi"
    )
    parser.add_argument("--sample-seconds", type=_positive_int, default=5)
    args = parser.parse_args(argv)
    try:
        job = render_job(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(job, indent=2))


if __name__ == "__main__":
    main()
