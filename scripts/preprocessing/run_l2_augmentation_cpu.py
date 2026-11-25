"""Render an offline CPU Job for L2 selection preparation or dataset cards."""

from __future__ import annotations

import argparse
import json
import re
from typing import TYPE_CHECKING, Any

from kube_jobs import storage
from kube_jobs.constants import TTL_SECONDS_AFTER_FINISHED
from kube_jobs.resolve_storage import resolve_storage
from kubernetes import client


if TYPE_CHECKING:
    from pathlib import PurePosixPath

if __package__:
    from scripts.preprocessing import run_l2_augmentation_a40 as shared
else:  # Support both direct script execution and python -m.
    import run_l2_augmentation_a40 as shared  # type: ignore[no-redef]


BOOTSTRAP = r"""set +x
set -euo pipefail
source_commit=$1
shift
export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=credential.https://gitlab.ics.muni.cz.helper
export GIT_CONFIG_VALUE_0='!f() { if [ "$1" = get ]; then printf "%s\n" "username=$VCS_USER" "password=$VCS_PASSWORD"; fi; }; f'
git clone --no-checkout https://github.com/RationAI/mammaprint.git /scratch/mammaprint
cd /scratch/mammaprint
git checkout --detach "$source_commit"
uv sync --frozen
unset VCS_USER VCS_PASSWORD GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
exec "$@"
"""


def _input(value: str) -> str:
    """Allow only mounted paths or the artifact URI forms used by this workflow."""
    if re.fullmatch(r"(?:runs|mlflow-artifacts):/[^\s]+", value):
        if ".." in value.split("/"):
            raise argparse.ArgumentTypeError("Artifact URI cannot contain '..'")
        return value
    path = shared._absolute_path(value)
    if shared.PROJECT not in path.parents and shared.DATA not in path.parents:
        raise argparse.ArgumentTypeError(
            "Inputs must be below /mnt/projects/mammaprint or /mnt/data"
        )
    return value


def _local_input(value: str) -> PurePosixPath:
    path = shared._absolute_path(value)
    _input(value)
    return path


def _validate(args: argparse.Namespace) -> dict[str, str]:
    if not args.context.strip() or not args.context.isprintable():
        raise ValueError("--context must name the intended cluster context")
    if shared.PROJECT not in args.output_root.parents:
        raise ValueError("--output-root must be below /mnt/projects/mammaprint")
    if args.output_root not in args.output.parents:
        raise ValueError("--output must be strictly below --output-root")
    if args.action == "prepare":
        if args.full_clean_val and (args.split != "val" or args.slides):
            raise ValueError("--full-clean-val requires --split val without --slides")
        if args.split == "val" and not (args.slides or args.full_clean_val):
            raise ValueError(
                "Validation preparation requires --slides or --full-clean-val"
            )
        if len(args.slides) != len(set(args.slides)):
            raise ValueError("--slides must be unique")
    else:
        if args.output.suffix not in (".yaml", ".yml"):
            raise ValueError("Card --output must be a YAML path")
        if bool(args.clean_train) != bool(args.fresh_val_manifest):
            raise ValueError(
                "--clean-train and --fresh-val-manifest must be supplied together"
            )
        if args.clean_train and (
            args.clean_test or ":" in args.clean_train or ":" in args.clean_val
        ):
            raise ValueError(
                "Fresh controls require local clean train/val paths and omit --clean-test"
            )
    shared._size_bytes(args.memory)
    scratch = shared._size_bytes(args.scratch_size)
    ephemeral = args.ephemeral_storage
    if ephemeral is None:
        ephemeral = (
            f"{(scratch + shared.SCRATCH_MARGIN) // shared.GIB}Gi"
            if scratch % shared.GIB == 0
            else f"{(scratch + shared.SCRATCH_MARGIN) // 1024**2}Mi"
        )
    if shared._size_bytes(ephemeral) < scratch + shared.SCRATCH_MARGIN:
        raise ValueError(
            "--ephemeral-storage must cover scratch plus 8Gi for logs and the container"
        )
    return {"cpu": str(args.cpu), "memory": args.memory, "ephemeral-storage": ephemeral}


def _worker(args: argparse.Namespace) -> list[str]:
    worker = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "preprocessing.l2_augmentation",
        args.action,
    ]
    names = (
        ("raw_tiling", "clean_uri", "data_mapping", "split")
        if args.action == "prepare"
        else (
            "manifest",
            "run_config",
            "augmented_uri",
            "clean_val",
            "clean_test",
            "clean_train",
            "fresh_val_manifest",
        )
    )
    for name in (*names, "output"):
        value = getattr(args, name)
        if value is not None:
            worker.extend((f"--{name.replace('_', '-')}", str(value)))
    if args.action == "prepare":
        if args.slides:
            worker.extend(("--slides", *args.slides))
        if args.full_clean_val:
            worker.append("--full-clean-val")
    return worker


def render_job(args: argparse.Namespace) -> dict[str, Any]:
    resources = _validate(args)
    mounts, volumes = resolve_storage([storage.secure.DATA, storage.secure.PROJECTS])
    with client.ApiClient() as api:
        volume_mounts = api.sanitize_for_serialization(mounts)
        volume_specs = api.sanitize_for_serialization(volumes)
    for mount in volume_mounts:
        mount["readOnly"] = True
        if mount["name"] == "projects":
            mount.update(
                mountPath=str(shared.PROJECT),
                subPath=str(shared.PROJECT.relative_to(shared.PROJECTS)),
            )
    for volume in volume_specs:
        if volume["name"] in ("data", "public-data"):
            volume["persistentVolumeClaim"]["readOnly"] = True
    volume_mounts.extend(
        [
            {
                "name": "projects",
                "mountPath": str(args.output_root),
                "subPath": str(args.output_root.relative_to(shared.PROJECTS)),
            },
            {"name": "scratch", "mountPath": "/scratch"},
        ]
    )
    volume_specs.append(
        {"name": "scratch", "emptyDir": {"sizeLimit": args.scratch_size}}
    )
    # Kubelet expands $(NAME) even within arguments Bash treats literally.
    command = [
        value.replace("$", "$$")
        for value in [BOOTSTRAP, args.job_name, args.source_commit, *_worker(args)]
    ]
    env = [
        entry
        for entry in shared._environment()
        if entry["name"] not in ("HF_TOKEN", "HF_HOME", "POD_UID")
    ]
    job: dict[str, Any] = {
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
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "volumes": volume_specs,
                    "containers": [
                        {
                            "name": f"l2-{args.action}",
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
                            "env": env,
                        }
                    ],
                }
            },
        },
    }
    if args.create_output_root:
        job["spec"]["template"]["spec"]["initContainers"] = [
            {
                "name": "create-output-root",
                "image": args.image,
                "command": [
                    "mkdir",
                    "-p",
                    "--",
                    str(args.output_root),
                    str(args.output_root / "outputs"),
                ],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "resources": {
                    "requests": {"cpu": "100m", "memory": "64Mi"},
                    "limits": {"cpu": "100m", "memory": "64Mi"},
                },
                "volumeMounts": [
                    {"name": "projects", "mountPath": str(shared.PROJECTS)}
                ],
            }
        ]
    return job


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "card"):
        stage = modes.add_parser(action, allow_abbrev=False)
        stage.add_argument("--context", required=True)
        stage.add_argument("--namespace", type=shared._dns_label, required=True)
        stage.add_argument("--job-name", type=shared._dns_label, required=True)
        stage.add_argument(
            "--source-commit",
            type=lambda sha: shared._hex_id(sha, 40),
            required=True,
            help="Full published worker commit SHA",
        )
        stage.add_argument(
            "--output-root",
            type=shared._absolute_path,
            required=True,
            help="Precreated writable project subtree",
        )
        stage.add_argument("--output", type=shared._absolute_path, required=True)
        stage.add_argument(
            "--create-output-root",
            action="store_true",
            help="Create the requested output subtree with a short init container",
        )
        stage.add_argument("--image", default=shared.IMAGE)
        stage.add_argument("--cpu", type=shared._positive_int, default=8)
        stage.add_argument("--memory", default="16Gi")
        stage.add_argument("--scratch-size", default="32Gi")
        stage.add_argument(
            "--ephemeral-storage", help="Defaults to --scratch-size plus 8Gi"
        )
        if action == "prepare":
            stage.add_argument("--raw-tiling", type=_input, required=True)
            stage.add_argument("--clean-uri", type=_input, required=True)
            stage.add_argument("--data-mapping", type=_local_input, required=True)
            stage.add_argument("--split", choices=("train", "val"), required=True)
            stage.add_argument(
                "--slides", nargs="+", type=shared._identifier, default=[]
            )
            stage.add_argument("--full-clean-val", action="store_true")
        else:
            stage.add_argument("--manifest", type=_local_input, required=True)
            stage.add_argument("--run-config", type=_local_input, required=True)
            stage.add_argument("--augmented-uri", type=_input, required=True)
            stage.add_argument("--clean-val", type=_input, required=True)
            stage.add_argument("--clean-test", type=_input)
            stage.add_argument("--clean-train", type=_input)
            stage.add_argument("--fresh-val-manifest", type=_local_input)
    args = parser.parse_args(argv)
    try:
        job = render_job(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(job, indent=2))


if __name__ == "__main__":
    main()
