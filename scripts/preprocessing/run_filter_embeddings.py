"""Render L1 preparation and publication jobs or launch the L2 through L5 filters."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from kube_jobs import storage, submit_job
from kube_jobs.constants import CLUSTER, HTTPS_PROXY, JOB_NAMESPACE, NO_PROXY
from kube_jobs.resolve_storage import resolve_storage
from kubernetes import client


USERNAME = "kissmi"
ROOT = Path(__file__).resolve().parents[2]
IMAGE = "cerit.io/rationai/base:2.0.6"
PUBLISHED_BASE = "e86923d6223a5b88ad9efcc9bd3da9db31fbbfd2"
L1_JOB_NAME = "mammaprint-l1-publish-cancer-splits"
L1_PREP_JOB_NAME = "mammaprint-l1-prepare-tissue-selection"
L1_TISSUE_JOB_NAME = "mammaprint-l1-publish-tissue-masks"
L1_TISSUE_TILING_URI = (
    "mlflow-artifacts:/3/43602e9f922d46f18dc06910b66a3be1/artifacts/mou_1_224_tissue"
)
L1_CANCER_MASKS_URI = (
    "mlflow-artifacts:/2/70b1c915966d43e3b5be46668075bd93/artifacts/heatmaps"
)
PUBLICATION_FILES = (
    "preprocessing/filter_embeddings.py",
    "preprocessing/embedded_mask_publication.py",
    "preprocessing/split_dataset.py",
)
PREPARATION_FILES = (
    "preprocessing/embedding_selection.py",
    "preprocessing/prepare_l1_tissue.py",
)
TISSUE_PUBLICATION_FILES = (
    *PUBLICATION_FILES,
    "preprocessing/embedding_selection.py",
    "preprocessing/publish_l1_tissue.py",
)
GIT_BRANCH = "feat/tiling-values"
TISSUE_EMBEDDING_URIS = {
    2: "mlflow-artifacts:/3/d70b72d842604ebe93d3fbd9e1dfd2f5/artifacts/embeddings",
    3: "mlflow-artifacts:/3/c513c87eac3d4792ba9757a2d7300eb6/artifacts/embeddings",
    4: "mlflow-artifacts:/3/ecfc0ecdba5b44d9becc5c9df1103a16/artifacts/embeddings",
    5: "mlflow-artifacts:/3/e4beda97fdab43f08e8b71fc5bea733c/artifacts/embeddings",
}
TISSUE_EMBEDDING_PATHS = {
    level: f"/mnt/projects/mammaprint/embeddings/l{level}_tissue_embed"
    for level in TISSUE_EMBEDDING_URIS
}
STRATEGY_CHOICES = ("epithelium_only", "or", "and")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level", type=int, choices=(1, *TISSUE_EMBEDDING_URIS), required=True
    )
    parser.add_argument("--tiling-uri")
    parser.add_argument("--embeddings-uri")
    parser.add_argument("--embeddings-path")
    parser.add_argument("--epithelium-threshold", type=float, default=0.25)
    parser.add_argument("--cancer-threshold", type=float, default=0.5)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGY_CHOICES,
        default=list(STRATEGY_CHOICES),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--upload-max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-l1-tissue",
        action="store_true",
        help="Prepare the scored L1 tissue selection for subsequent embedding",
    )
    parser.add_argument("--l1-source-run-id", help="Completed L1 A40 embedding run")
    parser.add_argument(
        "--tissue-preparation-run-id", help="Completed broad L1 tissue preparation"
    )
    parser.add_argument(
        "--tissue-embedding-run-id", help="Completed L1 missing-feature embedding run"
    )
    parser.add_argument("--source-commit", help="Local commit with the L1 worker")
    parser.add_argument("--publication-run-id", help="Resume a stopped L1 publication")
    parser.add_argument("--context", default=CLUSTER)
    parser.add_argument("--namespace", default=JOB_NAMESPACE)
    parser.add_argument("--job-name")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Create the rendered L1 job after approval",
    )
    args = parser.parse_args()
    if args.level == 1:
        if not args.l1_source_run_id or not args.source_commit:
            parser.error("L1 requires --l1-source-run-id and --source-commit")
        if args.dry_run and args.submit:
            parser.error("--dry-run and --submit are mutually exclusive")
        if (args.tissue_preparation_run_id is None) != (
            args.tissue_embedding_run_id is None
        ):
            parser.error(
                "--tissue-preparation-run-id and --tissue-embedding-run-id must be provided together"
            )
        if args.prepare_l1_tissue and args.tissue_preparation_run_id is not None:
            parser.error(
                "--prepare-l1-tissue cannot be combined with tissue publication"
            )
        if args.prepare_l1_tissue and args.publication_run_id is not None:
            parser.error("--publication-run-id applies only to L1 publication")
        if (
            args.tiling_uri
            or args.embeddings_uri
            or args.embeddings_path
            or args.output_dir
            or args.upload_only
            or args.upload_max_retries != 3
            or args.epithelium_threshold != 0.25
            or args.cancer_threshold != 0.5
            or tuple(args.strategies) != STRATEGY_CHOICES
        ):
            parser.error("L1 uses pinned sources and thresholds")
        if args.job_name is None:
            if args.prepare_l1_tissue:
                args.job_name = L1_PREP_JOB_NAME
            elif args.tissue_preparation_run_id is not None:
                args.job_name = L1_TISSUE_JOB_NAME
            else:
                args.job_name = L1_JOB_NAME
    else:
        if not args.tiling_uri:
            parser.error("L2 through L5 filtering requires --tiling-uri")
        if (
            args.prepare_l1_tissue
            or args.l1_source_run_id
            or args.tissue_preparation_run_id is not None
            or args.tissue_embedding_run_id is not None
            or args.source_commit
            or args.publication_run_id
            or args.submit
            or args.context != CLUSTER
            or args.namespace != JOB_NAMESPACE
            or args.job_name not in (None, L1_JOB_NAME)
        ):
            parser.error("L1 options require --level 1")
    return args


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "")


def _l1_script(commit: str, args: argparse.Namespace) -> str:
    command = ["uv", "run", "--frozen", "python", "-m"]
    files: tuple[str, ...]
    if args.prepare_l1_tissue:
        files = PREPARATION_FILES
        worker = "preprocessing/prepare_l1_tissue.py"
        command.extend(
            [
                "preprocessing.prepare_l1_tissue",
                "--tiling-uri",
                L1_TISSUE_TILING_URI,
                "--cancer-run-id",
                args.l1_source_run_id,
                "--cancer-masks-uri",
                L1_CANCER_MASKS_URI,
                "--epithelium-masks",
                "/mnt/projects/mammaprint/epithelium_onnx_masks",
                "--data-mapping",
                "/mnt/projects/mammaprint/data_mapping.csv",
                "--work-dir",
                "/scratch",
                "--workers",
                "8",
            ]
        )
    elif args.tissue_preparation_run_id is not None:
        files = TISSUE_PUBLICATION_FILES
        worker = "preprocessing/publish_l1_tissue.py"
        command.extend(
            [
                "preprocessing.publish_l1_tissue",
                "--preparation-run-id",
                args.tissue_preparation_run_id,
                "--embedding-run-id",
                args.tissue_embedding_run_id,
                "--cancer-run-id",
                args.l1_source_run_id,
                "--work-dir",
                "/scratch",
                "--data-mapping",
                "/mnt/projects/mammaprint/data_mapping.csv",
            ]
        )
    else:
        files = PUBLICATION_FILES
        worker = "preprocessing/embedded_mask_publication.py"
        command.extend(
            [
                "preprocessing.filter_embeddings",
                "--level",
                "1",
                "--l1-source-run-id",
                args.l1_source_run_id,
                "--work-dir",
                "/scratch",
                "--data-mapping",
                "/mnt/projects/mammaprint/data_mapping.csv",
            ]
        )
    if args.publication_run_id:
        command.extend(("--publication-run-id", args.publication_run_id))
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-renames",
            PUBLISHED_BASE,
            commit,
            "--",
            *files,
        ],
        cwd=ROOT,
    )
    if f"diff --git a/{worker} ".encode() not in diff:
        raise ValueError(f"{commit} lacks the committed L1 worker {worker}")
    overlay = base64.b64encode(gzip.compress(diff, mtime=0)).decode()
    return "\n".join(
        [
            "set -euo pipefail",
            'git config --global --replace-all url."https://${VCS_USER}:${VCS_PASSWORD}@gitlab.ics.muni.cz/".insteadOf "ssh://git@gitlab.ics.muni.cz/"',
            'git config --global --add url."https://${VCS_USER}:${VCS_PASSWORD}@gitlab.ics.muni.cz/".insteadOf "https://gitlab.ics.muni.cz/"',
            "git clone --no-checkout https://github.com/RationAI/mammaprint.git /scratch/mammaprint",
            "cd /scratch/mammaprint",
            f"git checkout --detach {PUBLISHED_BASE}",
            f"printf '%s' {shlex.quote(overlay)} | base64 --decode | gzip -dc | git apply",
            "uv sync --frozen",
            shlex.join(command),
        ]
    )


def _l1_environment() -> list[dict[str, Any]]:
    environment = {
        "MLFLOW_TRACKING_URI": "http://mlflow-s3.rationai-mlflow",
        "MLFLOW_TRACKING_USERNAME": USERNAME,
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "600",
        "HTTPS_PROXY": HTTPS_PROXY,
        "NO_PROXY": f"{NO_PROXY},mlflow-s3.rationai-mlflow",
        "TMPDIR": "/scratch",
        "UV_CACHE_DIR": "/scratch/uv-cache",
        "GIT_CONFIG_GLOBAL": "/scratch/gitconfig",
    }
    return [
        *({"name": name, "value": value} for name, value in environment.items()),
        *(
            {
                "name": name,
                "valueFrom": {
                    "secretKeyRef": {"name": "gitlab-deploy-token", "key": key}
                },
            }
            for name, key in (("VCS_USER", "username"), ("VCS_PASSWORD", "password"))
        ),
    ]


def _l1_job(args: argparse.Namespace) -> dict[str, Any]:
    commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{args.source_commit}^{{commit}}",
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    mounts, volumes = resolve_storage(storages=[storage.secure.PROJECTS])
    serialize = client.ApiClient().sanitize_for_serialization
    resources = {"cpu": "16", "memory": "64Gi"}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": args.job_name,
            "namespace": args.namespace,
            "labels": {"created_by": USERNAME},
            "annotations": {
                "mammaprint/source-commit": commit,
                "mammaprint/cluster-context": args.context,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "volumes": [
                        *(serialize(volume) for volume in volumes),
                        {"name": "scratch", "emptyDir": {"sizeLimit": "32Gi"}},
                    ],
                    "containers": [
                        {
                            "name": args.job_name,
                            "image": IMAGE,
                            "imagePullPolicy": "Always",
                            "command": [
                                "/bin/bash",
                                "-c",
                                _l1_script(commit, args),
                            ],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {"requests": resources, "limits": resources},
                            "volumeMounts": [
                                *(
                                    dict(serialize(mount), readOnly=True)
                                    for mount in mounts
                                ),
                                {"name": "scratch", "mountPath": "/scratch"},
                            ],
                            "env": _l1_environment(),
                        }
                    ],
                }
            },
        },
    }


def _legacy(args: argparse.Namespace) -> None:
    if args.upload_only and not args.output_dir:
        raise ValueError("--upload-only requires --output-dir")
    embeddings_uri = args.embeddings_uri or TISSUE_EMBEDDING_URIS[args.level]
    embeddings_path = args.embeddings_path or TISSUE_EMBEDDING_PATHS[args.level]
    strategies = tuple(dict.fromkeys(args.strategies))
    strategy_token = "-".join(
        {"epithelium_only": "epi", "or": "or", "and": "and"}[strategy]
        for strategy in strategies
    )
    job_name = (
        f"mammaprint-filter-embeddings-l{args.level}-"
        f"{strategy_token}-epi{_token(args.epithelium_threshold)}-"
        f"c{_token(args.cancer_threshold)}"
    )
    command = [
        "uv",
        "run",
        "-m",
        "preprocessing.filter_embeddings",
        "--level",
        str(args.level),
        "--tiling-uri",
        args.tiling_uri,
        "--embeddings-uri",
        embeddings_uri,
        "--embeddings-path",
        embeddings_path,
        "--epithelium-threshold",
        f"{args.epithelium_threshold:g}",
        "--cancer-threshold",
        f"{args.cancer_threshold:g}",
        "--strategies",
        *strategies,
        "--upload-max-retries",
        str(args.upload_max_retries),
    ]
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.upload_only:
        command.append("--upload-only")
    filter_command = shlex.join(command)

    if args.dry_run:
        print(f"Job: {job_name}")
        print(f"Command: {filter_command}")
        return

    submit_job(
        job_name=job_name,
        username=USERNAME,
        image=IMAGE,
        cpu=16,
        memory="64Gi",
        shm="16Gi",
        gpu=None,
        public=False,
        script=[
            "git clone https://github.com/rationAI/mammaprint workdir",
            "cd workdir",
            f"git checkout {GIT_BRANCH}",
            "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
            "uv sync --frozen",
            filter_command,
        ],
        storage=[storage.secure.PROJECTS],
    )


def main() -> None:
    args = _parse_args()
    if args.level != 1:
        _legacy(args)
        return
    manifest = json.dumps(_l1_job(args), indent=2)
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
