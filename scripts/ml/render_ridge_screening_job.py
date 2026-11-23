"""Render a CPU Kubernetes Job for one ridge screening command."""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit


OPERATIONS = ("cache", "sanity", "screen", "lock", "test")
SCREENS = ("levels", "multilevel", "statistics", "masks", "learning-curve", "all")
OUTPUT_MOUNT = PurePosixPath("/ridge-output")
REPOSITORY = "https://github.com/RationAI/mammaprint.git"


def _absolute_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.root != "/" or path == PurePosixPath("/") or ".." in path.parts:
        raise argparse.ArgumentTypeError("expected a normalized absolute pod path")
    return path


def _subpath(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise argparse.ArgumentTypeError("expected a relative PVC directory")
    return str(path)


def _resource_name(value: str) -> str:
    if len(value) > 63 or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", value):
        raise argparse.ArgumentTypeError("expected a Kubernetes DNS label")
    return value


def _worker_command(args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "scripts.ml.run_ridge_screening",
        args.operation,
        "--config",
        str(args.config),
        "--output",
        str(args.output),
        *(["--screen", args.screen] if args.screen else []),
        *(["--pilot-slides", str(args.pilot_slides)] if args.pilot_slides else []),
        *(["--dry-run"] if args.dry_run else []),
    ]


def _secret(name: str, secret_name: str, key: str) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {"secretKeyRef": {"name": secret_name, "key": key}},
    }


def render_job(args: argparse.Namespace) -> dict[str, Any]:
    worker = _worker_command(args)
    # Kubernetes expands $(NAME) in command and args before Bash or uv sees it.
    worker = [value.replace("$", "$$") for value in worker]
    resources = {"cpu": str(args.cpu), "memory": args.memory}
    env: list[dict[str, Any]] = [
        {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
        {"name": "MLFLOW_TRACKING_URI", "value": args.tracking_uri},
        {"name": "MLFLOW_HTTP_REQUEST_TIMEOUT", "value": "600"},
        {"name": "TMPDIR", "value": str(args.scratch)},
        {"name": "UV_CACHE_DIR", "value": str(args.scratch / "uv-cache")},
    ]
    if args.https_proxy:
        env.extend(
            [
                {"name": "HTTPS_PROXY", "value": args.https_proxy},
                {"name": "NO_PROXY", "value": args.no_proxy},
            ]
        )
    container: dict[str, Any] = {
        "name": args.job_name,
        "image": args.image,
        "imagePullPolicy": "IfNotPresent" if "@sha256:" in args.image else "Always",
        "workingDir": str(args.workdir),
        "command": worker,
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        },
        "resources": {"requests": resources, "limits": resources},
        "volumeMounts": [
            {"name": "input", "mountPath": str(args.input_mount), "readOnly": True},
            {
                "name": "results",
                "mountPath": str(OUTPUT_MOUNT),
                "subPath": args.output_subpath,
            },
            {"name": "scratch", "mountPath": str(args.scratch)},
        ],
        "env": env,
    }
    if args.env_secret:
        container["envFrom"] = [{"secretRef": {"name": args.env_secret}}]
    if args.source_commit:
        container["workingDir"] = str(args.scratch)
        container["command"] = ["/bin/bash", "-ec"]
        script = ["export GIT_TERMINAL_PROMPT=0"]
        if args.gitlab_secret:
            env.extend(
                (
                    _secret("VCS_USER", args.gitlab_secret, "username"),
                    _secret("VCS_PASSWORD", args.gitlab_secret, "password"),
                )
            )
            script.extend(
                (
                    "export GIT_CONFIG_COUNT=1",
                    "export GIT_CONFIG_KEY_0=credential.https://gitlab.ics.muni.cz.helper",
                    'export GIT_CONFIG_VALUE_0=\'!f() { if [ "$1" = get ]; then '
                    'printf "%s\\n" "username=$VCS_USER" "password=$VCS_PASSWORD"; '
                    "fi; }; f'",
                )
            )
        script.extend(
            (
                f'git clone --no-checkout {REPOSITORY} "$TMPDIR/repo"',
                'git -C "$TMPDIR/repo" checkout --detach "$1"',
                "shift",
                'cd "$TMPDIR/repo"',
                'exec "$@"',
            )
        )
        container["args"] = [
            "\n".join(script),
            "ridge-job",
            args.source_commit,
            *worker,
        ]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": args.job_name,
            "namespace": args.namespace,
            "annotations": {"mammaprint.rationai.org/target-context": args.context},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "volumes": [
                        {
                            "name": "input",
                            "persistentVolumeClaim": {
                                "claimName": args.input_claim,
                                "readOnly": True,
                            },
                        },
                        {
                            "name": "results",
                            "persistentVolumeClaim": {"claimName": args.output_claim},
                        },
                        {
                            "name": "scratch",
                            "emptyDir": {"sizeLimit": args.scratch_size},
                        },
                    ],
                    "containers": [container],
                }
            },
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=OPERATIONS)
    parser.add_argument("--context", required=True, help="Intended cluster context")
    parser.add_argument("--namespace", required=True, type=_resource_name)
    parser.add_argument("--job-name", required=True, type=_resource_name)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-commit", help="Full Git SHA from the published repo")
    parser.add_argument(
        "--workdir", type=_absolute_path, default="/workspace/mammaprint"
    )
    parser.add_argument("--config", required=True, type=_absolute_path)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--input-mount", type=_absolute_path, default="/mnt/projects")
    parser.add_argument("--input-claim", default="projects-rw")
    parser.add_argument("--output-claim", default="projects-rw")
    parser.add_argument("--output-subpath", required=True, type=_subpath)
    parser.add_argument("--scratch", type=_absolute_path, default="/scratch")
    parser.add_argument("--scratch-size", default="64Gi")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--memory", default="16Gi")
    parser.add_argument("--tracking-uri", default="http://mlflow-s3.rationai-mlflow")
    parser.add_argument(
        "--env-secret", help="Existing Secret with MLflow or input env vars"
    )
    parser.add_argument(
        "--gitlab-secret", help="Existing deploy Secret with username/password"
    )
    parser.add_argument("--https-proxy", default="http://proxy.ics.muni.cz:3128")
    parser.add_argument(
        "--no-proxy", default="mlflow-s3.rationai-mlflow,svc.cluster.local"
    )
    parser.add_argument("--screen", choices=SCREENS)
    parser.add_argument(
        "--pilot-slides", type=int, help="Size of the fixed development pilot roster"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Pass --dry-run to the CLI"
    )
    args = parser.parse_args(argv)

    if not args.context.strip():
        parser.error("--context must name the intended cluster")
    if not args.image or re.search(r"\s", args.image):
        parser.error("--image must be an OCI image reference")
    if args.source_commit and not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be a full lowercase Git SHA")
    if ("@" in args.image or not args.source_commit) and not re.fullmatch(
        r"[^@]+@sha256:[0-9a-f]{64}", args.image
    ):
        parser.error("use an image digest or provide --source-commit")
    if args.gitlab_secret and not args.source_commit:
        parser.error("--gitlab-secret requires --source-commit")
    if args.cpu <= 0:
        parser.error("--cpu must be positive")
    if args.screen and args.operation != "screen":
        parser.error("--screen is valid only for the screen operation")
    if args.pilot_slides is not None and (
        args.operation != "cache" or args.pilot_slides <= 0
    ):
        parser.error("--pilot-slides requires a positive count and the cache operation")
    if args.config == args.input_mount or not args.config.is_relative_to(
        args.input_mount
    ):
        parser.error("--config must be a file under --input-mount")
    if not args.output.is_relative_to(OUTPUT_MOUNT):
        parser.error(f"--output must be under {OUTPUT_MOUNT}")
    mounts = (args.input_mount, OUTPUT_MOUNT, args.scratch)
    if any(
        left.is_relative_to(right) or right.is_relative_to(left)
        for left, right in combinations(mounts, 2)
    ):
        parser.error("input, output, and scratch mounts must use disjoint paths")
    for uri in (args.tracking_uri, args.https_proxy):
        if uri and urlsplit(uri).username is not None:
            parser.error("provide URI credentials through cluster Secrets")

    print(json.dumps(render_job(args), indent=2))


if __name__ == "__main__":
    main()
