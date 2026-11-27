"""Render an offline Kubernetes manifest for independent seeds on one GPU."""

import argparse
import base64
import gzip
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, get_args

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import HydraException
from kube_jobs import storage
from kube_jobs.constants import HTTPS_PROXY, NO_PROXY, TTL_SECONDS_AFTER_FINISHED
from kube_jobs.resolve_gpu import GPUConfig, JobGPUType, resolve_gpu
from kube_jobs.resolve_storage import resolve_storage
from kubernetes import client
from kubernetes.utils.quantity import parse_quantity

from scripts.ml.run_training_batch import BatchSpec, identifier, validate_spec


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "https://github.com/RationAI/mammaprint.git"
PROJECTS = PurePosixPath("/mnt/projects")
SCRATCH = "/training-scratch"
OVERLAY_PATHS = (
    "scripts/ml/run_training_batch.py",
    "scripts/ml/training_batch_resources.py",
)
EMBEDDING_LOADER_PATH = "ml/data/datasets/_sources.py"
ANNOTATION = "mammaprint.rationai.org/"
TAGS = ("batch", "run", "attempt", "execution")
OWNED_KEYS = (
    "seed",
    "hydra",
    "metadata.run_name",
    "logger.run_name",
    "logger.run_id",
    "trainer.accelerator",
    "trainer.devices",
    "trainer.default_root_dir",
    "trainer.num_nodes",
    "trainer.strategy",
    "trainer.callbacks.model_checkpoint.dirpath",
    "logger.tracking_uri",
    *(f"logger.tags.training_{tag}_id" for tag in TAGS),
    "logger.tags.training_source_commit",
    "logger.tags.training_recipe_commit",
)
BOOTSTRAP = rf"""set -euo pipefail
WORK={SCRATCH}/repo
export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=credential.https://gitlab.ics.muni.cz.helper
export GIT_CONFIG_VALUE_0='!f() {{ if [ "$1" = get ]; then printf "%s\n" "username=$VCS_USER" "password=$VCS_PASSWORD"; fi; }}; f'
git clone --no-checkout {REPOSITORY} "$WORK"
git -C "$WORK" checkout --detach "$1"
cd "$WORK"
if [ -n "$4" ]; then
    printf '%s' "$4" | base64 --decode | gzip -dc | git apply
fi
printf '%s\n' "$2" > {SCRATCH}/spec.json
uv sync --frozen
unset VCS_USER VCS_PASSWORD GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
.venv/bin/python -m scripts.ml.training_batch_resources --probe > {SCRATCH}/gpu.json
cat {SCRATCH}/gpu.json
exec .venv/bin/python -m scripts.ml.run_training_batch --spec {SCRATCH}/spec.json \
    --output-root "$3" --scratch-root {SCRATCH}/fits --attempt-id "$5" \
    --execution-id "$TRAIN_POD_UID" --gpu-info {SCRATCH}/gpu.json
"""


def git_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("Expected a full lowercase Git SHA")
    return value


def dns_label(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value):
        raise argparse.ArgumentTypeError("Expected a Kubernetes DNS label")
    return value


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def projects_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if PROJECTS not in path.parents or ".." in path.parts:
        raise argparse.ArgumentTypeError("Expected a subdirectory of /mnt/projects")
    return path


def validate_override(value: str) -> None:
    parsed = OverridesParser.create().parse_override(value)
    key = parsed.key_or_group.replace("/", ".")
    keys = [key] if not parsed.package else [key, parsed.package]
    if parsed.is_sweep_override() or any(
        key == owned or key.startswith(f"{owned}.") or owned.startswith(f"{key}.")
        for key in keys
        for owned in OWNED_KEYS
    ):
        raise ValueError(f"Batch-owned key or sweep in override: {value}")
    if key == "checkpoint" and parsed.value() is not None:
        checkpoint = parsed.value()
        if not isinstance(checkpoint, str) or not checkpoint.startswith(
            ("/", "mlflow-artifacts:/")
        ):
            raise ValueError(
                "An explicit checkpoint must use an absolute path or mlflow-artifacts:/ URI"
            )


def seed_spec(args: argparse.Namespace) -> BatchSpec:
    if not re.fullmatch(r"[a-zA-Z0-9_/-]+", args.experiment):
        raise ValueError("Expected a Hydra experiment config path")
    for override in args.override:
        validate_override(override)
    if any(not 0 <= seed <= 2**32 - 1 for seed in args.seeds):
        raise ValueError("Seeds must be unsigned 32-bit integers")

    common = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "ml.train",
        f"+experiment={args.experiment}",
        *args.override,
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "hydra.run.dir=${oc.env:TRAIN_RUN_DIR}",
        "hydra.job.chdir=true",
        "++trainer.default_root_dir=${oc.env:TRAIN_RUN_DIR}",
        "metadata.run_name=${oc.env:TRAIN_BATCH_ID}-${oc.env:TRAIN_RUN_ID}",
        f"+logger.tags.training_source_commit={args.source_commit}",
        f"+logger.tags.training_recipe_commit={args.overlay_base or args.source_commit}",
        *(
            f"+logger.tags.training_{tag}_id=${{oc.env:TRAIN_{tag.upper()}_ID}}"
            for tag in TAGS
        ),
    ]
    return validate_spec(
        {
            "version": 1,
            "batch_id": args.batch_id,
            "max_parallel": args.max_parallel,
            "runs": [
                {"id": f"seed-{seed}", "argv": [*common, f"seed={seed}"]}
                for seed in args.seeds
            ],
        }
    )


def committed_overlay(
    base: str, source: str, *, include_embedding_loader: bool = False
) -> str:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, source], cwd=ROOT, check=True
    )
    paths = (
        (*OVERLAY_PATHS, EMBEDDING_LOADER_PATH)
        if include_embedding_loader
        else OVERLAY_PATHS
    )
    patch = subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            "--no-ext-diff",
            base,
            source,
            "--",
            *paths,
        ],
        cwd=ROOT,
    )
    if not patch:
        raise ValueError("The source and overlay base contain no selected changes")
    return base64.b64encode(gzip.compress(patch, mtime=0)).decode("ascii")


def secret(name: str, key: str) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {"secretKeyRef": {"name": "gitlab-deploy-token", "key": key}},
    }


def render_job(args: argparse.Namespace) -> dict[str, Any]:
    if args.overlay_embedding_loader and not args.overlay_base:
        raise ValueError("--overlay-embedding-loader requires --overlay-base")
    spec = seed_spec(args)
    gpu = (
        GPUConfig(limits={"nvidia.com/mig-2g.24gb": 1})
        if args.gpu == "mig-2g.24gb"
        else resolve_gpu(args.gpu)
    )
    mounts, volumes = resolve_storage([storage.secure.DATA, storage.secure.PROJECTS])
    annotations = {
        ANNOTATION + "target-context": args.context,
        ANNOTATION + "source-commit": args.source_commit,
        ANNOTATION + "batch-id": spec["batch_id"],
    }
    overlay = ""
    if args.overlay_base:
        annotations[ANNOTATION + "overlay-base"] = args.overlay_base
        overlay = committed_overlay(
            args.overlay_base,
            args.source_commit,
            include_embedding_loader=args.overlay_embedding_loader,
        )
        if args.overlay_embedding_loader:
            annotations[ANNOTATION + "overlay-embedding-loader"] = "true"

    worker_args = [
        BOOTSTRAP,
        "training-batch",
        args.overlay_base or args.source_commit,
        json.dumps(spec, separators=(",", ":"), ensure_ascii=True),
        str(args.output_root),
        overlay,
        args.job_name,
    ]
    # Kubelet expands $(NAME) before Bash processes arguments.
    worker_args = [value.replace("$", "$$") for value in worker_args]
    if any(len(value.encode()) >= 96 * 1024 for value in worker_args):
        raise ValueError(
            "A bootstrap argument is too large. Use a smaller batch or a published source."
        )

    resources = {"cpu": args.cpu, "memory": args.memory, **gpu.limits}
    if args.ephemeral_storage is not None:
        scratch_size = parse_quantity(args.scratch_size)
        ephemeral_storage = parse_quantity(args.ephemeral_storage)
        if (
            not scratch_size.is_finite()
            or not ephemeral_storage.is_finite()
            or scratch_size <= 0
            or ephemeral_storage < scratch_size
        ):
            raise ValueError(
                "Ephemeral storage must cover the positive scratch volume size"
            )
        resources["ephemeral-storage"] = args.ephemeral_storage
    environment = {
        "MLFLOW_TRACKING_URI": "http://mlflow-s3.rationai-mlflow",
        "MLFLOW_TRACKING_USERNAME": args.username,
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "600",
        "HYDRA_FULL_ERROR": "1",
        "HTTPS_PROXY": HTTPS_PROXY,
        "NO_PROXY": f"{NO_PROXY},mlflow-s3.rationai-mlflow",
        "UV_CACHE_DIR": f"{SCRATCH}/uv-cache",
        "MPLCONFIGDIR": f"{SCRATCH}/matplotlib",
    }
    if args.threads_per_fit:
        environment.update(
            {
                name: str(args.threads_per_fit)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            }
        )

    with client.ApiClient() as api:
        storage_mounts = api.sanitize_for_serialization(mounts)
        storage_volumes = api.sanitize_for_serialization(volumes)
    pod = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 20,
        "automountServiceAccountToken": False,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "volumes": [
            *storage_volumes,
            {"name": "scratch", "emptyDir": {"sizeLimit": args.scratch_size}},
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": args.shm}},
        ],
        "containers": [
            {
                "name": "training-batch",
                "image": args.image,
                "imagePullPolicy": "IfNotPresent"
                if "@sha256:" in args.image
                else "Always",
                "command": ["/bin/bash", "-c"],
                "args": worker_args,
                "resources": {"requests": resources, "limits": resources},
                "env": [
                    *(
                        {"name": key, "value": value}
                        for key, value in environment.items()
                    ),
                    secret("VCS_USER", "username"),
                    secret("VCS_PASSWORD", "password"),
                    {
                        "name": "TRAIN_POD_UID",
                        "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                    },
                ],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [
                    *storage_mounts,
                    {"name": "scratch", "mountPath": SCRATCH},
                    {"name": "shm", "mountPath": "/dev/shm"},
                ],
            }
        ],
    }
    if gpu.node_selector:
        pod["nodeSelector"] = gpu.node_selector
    job_spec = {
        "backoffLimit": 0,
        "ttlSecondsAfterFinished": TTL_SECONDS_AFTER_FINISHED,
        "template": {"metadata": {"labels": {"app": args.job_name}}, "spec": pod},
    }
    if args.active_deadline_seconds:
        job_spec["activeDeadlineSeconds"] = args.active_deadline_seconds
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": args.job_name,
            "namespace": args.namespace,
            "labels": {"created_by": args.username},
            "annotations": annotations,
        },
        "spec": job_spec,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True, type=dns_label)
    parser.add_argument("--job-name", required=True, type=dns_label)
    parser.add_argument("--batch-id", required=True, type=identifier)
    parser.add_argument("--source-commit", required=True, type=git_sha)
    parser.add_argument(
        "--overlay-base",
        type=git_sha,
        help="Published training recipe with a local launcher overlay",
    )
    parser.add_argument(
        "--overlay-embedding-loader",
        action="store_true",
        help="Include committed MLflow download helper changes with the overlay",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--gpu", required=True, choices=(*get_args(JobGPUType.__value__), "mig-2g.24gb")
    )
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--shm", required=True)
    parser.add_argument("--scratch-size", default="128Gi")
    parser.add_argument("--ephemeral-storage", help="Pod local disk request and limit")
    parser.add_argument("--output-root", required=True, type=projects_path)
    parser.add_argument("--username", default="kissmi")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-parallel", type=positive_int, default=1)
    parser.add_argument("--threads-per-fit", type=positive_int)
    parser.add_argument("--active-deadline-seconds", type=positive_int)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        print(json.dumps(render_job(args), indent=2))
    except (HydraException, ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
