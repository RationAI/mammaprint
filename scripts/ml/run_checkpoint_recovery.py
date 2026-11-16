"""Plan one bounded checkpoint recovery job; only --submit contacts Kubernetes.

Ordinary Python planning is stdlib-only, offline and read-only. The generated job
trains into a linked MLflow run, then backs up and replaces the selected original
checkpoint after validation and hash checks. No sweep is inferred from a sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shlex
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


# Reuse the existing fixed-target validator/submission contract without adding
# Kubernetes, MLflow or the application's training imports to offline planning.
_spec = importlib.util.spec_from_file_location(
    "_checkpoint_cluster", Path(__file__).with_name("run_concept_explainability.py")
)
cluster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cluster)


def _source_id(source: str, tracking_uri: str) -> str:
    """Validate source identity offline; resolved metadata is checked by the core."""
    cluster._plain(source, "source")
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", source):
        return source
    parsed = urlsplit(source)
    match = re.fullmatch(
        r"/?experiments/(\d+)/runs/([A-Za-z0-9_-]{1,128})"
        r"(?:/(?:model-metrics|artifacts|overview|system-metrics))?/?",
        parsed.fragment or parsed.path,
    )
    if (
        not match
        or parsed.query
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "--source requires a safe run ID or credentials-free MLflow experiment/run URL"
        )
    base_path = parsed.path if parsed.fragment else ""
    source_endpoint = urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), base_path.rstrip("/"), "", "")
    )
    target = urlsplit(tracking_uri)
    target_endpoint = urlunsplit(
        (target.scheme.lower(), target.netloc.lower(), target.path.rstrip("/"), "", "")
    )
    if source_endpoint != target_endpoint:
        raise ValueError("--source URL and --tracking-uri endpoint disagree")
    return match[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="One existing MLflow run URL or run ID."
    )
    parser.add_argument(
        "--git-sha",
        required=True,
        help="Exact full reviewed implementation SHA, already pushed.",
    )
    parser.add_argument(
        "--git-ref",
        help="Optional reachable remote ref; checkout remains pinned to --git-sha.",
    )
    parser.add_argument(
        "--training-sha",
        help="Exact historical SHA only when the source Git tag is absent.",
    )
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument(
        "--output-root",
        required=True,
        help="Fresh absolute directory beneath /mnt/projects; missing parents are created.",
    )
    parser.add_argument(
        "--monitor",
        help="Optional changed selection rule: exact validation metric key.",
    )
    parser.add_argument("--mode", choices=("min", "max"))
    parser.add_argument("--job-prefix", default="mammaprint-recovery")
    parser.add_argument("--username", default="kissmi")
    parser.add_argument("--image", default=cluster.IMAGE)
    parser.add_argument("--gpu", choices=("H100", "A100", "A40"), default="H100")
    parser.add_argument("--cpu", type=cluster._positive, default=16)
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--shm", default="16Gi")
    parser.add_argument(
        "--context",
        default=cluster.CONTEXT,
        help="Expected fixed helper context, not routing.",
    )
    parser.add_argument(
        "--namespace",
        default=cluster.NAMESPACE,
        help="Expected fixed helper namespace, not routing.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Create the job; requires explicit cluster submission authorization.",
    )
    parser.add_argument(
        "--replace-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Back up and rewrite this selected run's best checkpoint after successful replay (default).",
    )
    args = parser.parse_args(argv)
    # Delegate shared resource, URI, Git pin/ref, output and label checks to the
    # existing launcher. Its concept-only arguments are never used in this plan.
    common = ["--source", f"recovery:{args.git_sha}"]
    for name in (
        "git_sha",
        "tracking_uri",
        "output_root",
        "job_prefix",
        "username",
        "image",
        "gpu",
        "cpu",
        "memory",
        "shm",
        "context",
        "namespace",
    ):
        common.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    common.extend(["--ray-cpus", str(args.cpu)])
    if args.git_ref is not None:
        common.extend(["--git-ref", args.git_ref])
    validated = cluster.parse_args(common)
    args.git_sha = validated.git_sha
    args.output_root = validated.output_root
    try:
        endpoint = urlsplit(args.tracking_uri)
        if (
            "%" in endpoint.netloc
            or "\\" in args.tracking_uri
            or any(p in {".", ".."} for p in endpoint.path.split("/"))
        ):
            raise ValueError("--tracking-uri contains an invalid endpoint")
        args.source_run_id = _source_id(args.source, args.tracking_uri)
        if args.training_sha is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{40}", args.training_sha):
                raise ValueError(
                    "--training-sha requires an exact 40-character Git SHA"
                )
            args.training_sha = args.training_sha.lower()
        if (args.monitor is None) != (args.mode is None):
            raise ValueError("--monitor and --mode must be supplied together")
        if args.monitor is not None:
            cluster._plain(args.monitor, "monitor")
            if not re.fullmatch(r"valid/[A-Za-z0-9_.-]+", args.monitor):
                raise ValueError("--monitor must name an exact valid/... metric")
    except ValueError as error:
        parser.error(str(error))
    return args


def _script(args: argparse.Namespace, output: str) -> list[str]:
    script = [
        "set -euo pipefail",
        "umask 077",
        shlex.join(["git", "clone", "--no-checkout", cluster.REPOSITORY, "workdir"]),
        "cd workdir",
    ]
    if args.git_ref:
        script.append(shlex.join(["git", "fetch", "origin", args.git_ref]))
    script.extend(
        [
            f"git cat-file -e {args.git_sha}^{{commit}} 2>/dev/null || git fetch origin {args.git_sha}",
            f"git checkout --detach {args.git_sha}",
            f'test "$(git rev-parse HEAD)" = {args.git_sha}',
        ]
    )
    if args.training_sha:
        script.extend(
            [
                f"git cat-file -e {args.training_sha}^{{commit}} 2>/dev/null || git fetch origin {args.training_sha}",
                f"git cat-file -e {args.training_sha}^{{commit}}",
            ]
        )
    script.extend(
        [
            shlex.join(["mkdir", "-p", "--", str(Path(args.output_root).parent)]),
            shlex.join(["mkdir", "--", args.output_root]),
            "uv sync --frozen",
            shlex.join(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "python",
                    "-c",
                    "import torch; assert torch.cuda.is_available(), 'CUDA is required for recovery training'; print('CUDA', torch.version.cuda, torch.cuda.get_device_name(0))",
                ]
            ),
            "unset MLFLOW_RUN_ID",
            f"export MLFLOW_TRACKING_URI={shlex.quote(args.tracking_uri)}",
        ]
    )
    command = [
        "uv",
        "run",
        "--frozen",
        "-m",
        "ml.recover_checkpoint",
        args.source,
        "--tracking-uri",
        args.tracking_uri,
        "--implementation-sha",
        args.git_sha,
        "--output-root",
        output,
        "--execute",
    ]
    if args.training_sha:
        command.extend(["--training-sha", args.training_sha])
    if args.monitor:
        command.extend(["--monitor", args.monitor, "--mode", args.mode])
    if args.replace_original:
        command.append("--replace-original")
    script.append(shlex.join(command))
    return script


def build_plan(args: argparse.Namespace) -> dict:
    output = args.output_root + "/replay"
    identity = json.dumps(
        [
            args.source,
            args.training_sha,
            args.git_sha,
            output,
            args.monitor,
            args.mode,
            args.replace_original,
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return {
        "mode": "submit" if args.submit else "plan",
        "remote_verified": False,
        "repository": cluster.REPOSITORY,
        "image": args.image,
        "username": args.username,
        "target": {
            "context": args.context,
            "namespace": args.namespace,
            "helper_context": cluster.CONTEXT,
            "helper_namespace": cluster.NAMESPACE,
            "routing": "fixed private kube_jobs helper; flags are expectations",
        },
        "resources": {
            "gpu": args.gpu,
            "gpu_limit": 1,
            "cpu": args.cpu,
            "memory": args.memory,
            "shm": args.shm,
        },
        "execution": {
            "action": "bounded seeded retraining; fit train/validation only",
            "selection": {
                "monitor": args.monitor,
                "mode": args.mode,
                "default": "original ModelCheckpoint monitor/mode",
            },
            "mlflow": (
                "new linked recovery run, then backed-up, hash-guarded original checkpoint replacement"
                if args.replace_original
                else "new linked recovery run only; original unchanged"
            ),
            "replace_original": args.replace_original,
            "historical_object": "explicit pin fetched before core; otherwise core resolves/fetches source tag after identity checks",
        },
        "unverified": [
            "remote Git reachability of implementation and historical source pins",
            "source metadata/config/history, data bytes/mapping and dependency compatibility",
            "image CUDA/runtime, cluster/GPU capacity and resource sufficiency",
            "output parent availability and fresh root; original score reproducibility",
        ],
        "jobs": [
            {
                "job_name": f"{args.job_prefix}-{digest}",
                "source": args.source,
                "source_run_id": args.source_run_id,
                "training_sha": args.training_sha,
                "implementation_sha": args.git_sha,
                "implementation_ref": args.git_ref,
                "output_dir": output,
                "script": _script(args, output),
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    print(json.dumps(plan, indent=2))
    if args.submit:
        cluster.submit_plan(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
