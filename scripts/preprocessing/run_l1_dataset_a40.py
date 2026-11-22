"""Render one A40 L1 dataset job. Submission requires explicit project approval."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from kube_jobs import storage
from kube_jobs.constants import CLUSTER, HTTPS_PROXY, JOB_NAMESPACE, NO_PROXY
from kube_jobs.resolve_gpu import resolve_gpu
from kube_jobs.resolve_storage import resolve_storage
from kubernetes import client


ROOT = Path(__file__).resolve().parents[2]
BASE = "e86923d6223a5b88ad9efcc9bd3da9db31fbbfd2"
IMAGE = "cerit.io/rationai/base:2.0.6"
TILING = "mlflow-artifacts:/3/7651a68400f6484db36d3626b1d64e83/artifacts/mou_1_224_cancer_mask"
REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
FILES = (
    "preprocessing/embedding_selection.py",
    "preprocessing/embeddings.py",
    "preprocessing/resolution_embeddings.py",
    "preprocessing/embedding_artifacts.py",
    "preprocessing/create_l1_dataset.py",
)


def _environment() -> list[dict[str, Any]]:
    def secret(name: str, secret_name: str, key: str) -> dict[str, Any]:
        return {
            "name": name,
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": key}},
        }

    return [
        {
            "name": "MLFLOW_TRACKING_URI",
            "value": "http://mlflow-s3.rationai-mlflow",
        },
        {"name": "MLFLOW_TRACKING_USERNAME", "value": "kissmi"},
        {"name": "MLFLOW_HTTP_REQUEST_TIMEOUT", "value": "600"},
        {"name": "HTTPS_PROXY", "value": HTTPS_PROXY},
        {"name": "NO_PROXY", "value": f"{NO_PROXY},mlflow-s3.rationai-mlflow"},
        {"name": "TMPDIR", "value": "/scratch"},
        {"name": "HF_HOME", "value": "/scratch/huggingface"},
        {"name": "UV_CACHE_DIR", "value": "/scratch/uv-cache"},
        {"name": "GIT_CONFIG_GLOBAL", "value": "/scratch/gitconfig"},
        secret("HF_TOKEN", "huggingface-secret", "token"),
        secret("VCS_USER", "gitlab-deploy-token", "username"),
        secret("VCS_PASSWORD", "gitlab-deploy-token", "password"),
    ]


def _script(commit: str, run_id: str | None) -> str:
    diff = subprocess.check_output(
        ["git", "diff", "--binary", BASE, commit, "--", *FILES], cwd=ROOT
    )
    if not diff or b"create_l1_dataset.py" not in diff:
        raise ValueError(f"{commit} lacks the committed L1 dataset worker")
    overlay = base64.b64encode(gzip.compress(diff, mtime=0)).decode()
    command = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "preprocessing.create_l1_dataset",
        "--tiling",
        TILING,
        "--data-mapping",
        "/mnt/projects/mammaprint/data_mapping.csv",
        "--epithelium-masks",
        "/mnt/projects/mammaprint/epithelium_onnx_masks",
        "--work-dir",
        "/scratch/dataset",
        "--encoder-revision",
        REVISION,
        "--runtime-id",
        f"{IMAGE}@git:{commit}",
        "--batch-size",
        "1024",
        "--workers",
        "8",
    ]
    if run_id:
        command.extend(("--run-id", run_id))
    return "\n".join(
        [
            "set -euo pipefail",
            'git config --global --replace-all url."https://${VCS_USER}:${VCS_PASSWORD}@gitlab.ics.muni.cz/".insteadOf "ssh://git@gitlab.ics.muni.cz/"',
            'git config --global --add url."https://${VCS_USER}:${VCS_PASSWORD}@gitlab.ics.muni.cz/".insteadOf "https://gitlab.ics.muni.cz/"',
            "git clone https://github.com/RationAI/mammaprint.git /scratch/mammaprint",
            "cd /scratch/mammaprint",
            f"git checkout --detach {shlex.quote(BASE)}",
            f"printf '%s' {shlex.quote(overlay)} | base64 --decode | gzip -dc | git apply",
            "uv sync --frozen",
            "nvidia-smi --query-gpu=name,memory.total --format=csv",
            "uv run --frozen python -c "
            '\'import torch; assert torch.cuda.is_available(), "PyTorch CUDA is unavailable"; '
            'assert "A40" in torch.cuda.get_device_name(0), "Expected an A40"\'',
            shlex.join(command),
        ]
    )


def job(args: argparse.Namespace) -> dict[str, Any]:
    gpu = resolve_gpu("A40")
    mounts, volumes = resolve_storage(
        storages=[storage.secure.DATA, storage.secure.PROJECTS]
    )
    serialize = client.ApiClient().sanitize_for_serialization
    volume_mounts = [dict(serialize(mount), readOnly=True) for mount in mounts]
    volume_mounts.extend(
        [
            {"name": "shm", "mountPath": "/dev/shm"},
            {"name": "scratch", "mountPath": "/scratch"},
        ]
    )
    volume_specs = [serialize(volume) for volume in volumes]
    volume_specs.extend(
        [
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "12Gi"}},
            {"name": "scratch", "emptyDir": {"sizeLimit": "32Gi"}},
        ]
    )
    resources = {"cpu": "24", "memory": "64Gi", **gpu.limits}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": args.job_name,
            "namespace": args.namespace,
            "labels": {"created_by": "kissmi"},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
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
                            "name": args.job_name,
                            "image": IMAGE,
                            "imagePullPolicy": "Always",
                            "command": [
                                "/bin/bash",
                                "-c",
                                _script(args.source_commit, args.run_id),
                            ],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {"requests": resources, "limits": resources},
                            "volumeMounts": volume_mounts,
                            "env": _environment(),
                        }
                    ],
                }
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default=CLUSTER)
    parser.add_argument("--namespace", default=JOB_NAMESPACE)
    parser.add_argument("--job-name", default="mammaprint-l1-embed-a40")
    parser.add_argument(
        "--source-commit", required=True, help="Local committed code to package"
    )
    parser.add_argument("--run-id", help="Resume a stopped job from its MLflow run ID")
    parser.add_argument(
        "--submit", action="store_true", help="Create this job after explicit approval"
    )
    args = parser.parse_args()
    args.source_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.source_commit}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    manifest = json.dumps(job(args), indent=2)
    if args.submit:
        subprocess.run(
            [
                "kubectl",
                "--context",
                args.context,
                "--namespace",
                args.namespace,
                "create",
                "-f",
                "-",
            ],
            input=manifest,
            text=True,
            check=True,
        )
    else:
        print(manifest)


if __name__ == "__main__":
    main()
