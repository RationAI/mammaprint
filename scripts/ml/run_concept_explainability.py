"""Plan pinned deployed-metric concept jobs; only --submit contacts Kubernetes.

Run with ordinary Python for offline JSON planning. Submission additionally needs
the established kube_jobs helper and Kubernetes client in the local environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import shlex
from pathlib import PurePosixPath
from urllib.parse import urlsplit


CONTEXT = "kubas-cluster"
NAMESPACE = "rationai-jobs-ns"
REPOSITORY = "https://github.com/RationAI/mammaprint.git"
IMAGE = "cerit.io/rationai/base:2.0.6"


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _quantity_bytes(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)(Mi|Gi)", value)
    if not match:
        raise ValueError("memory and shm must be positive whole Mi or Gi quantities")
    return int(match[1]) * (1024 ** (2 if match[2] == "Mi" else 3))


def _plain(value: str, name: str) -> None:
    if not value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"{name} must be nonempty and contain no control characters")
    if "${" in value:
        raise ValueError(f"{name} cannot contain Hydra interpolation (${{...}})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, metavar="RUN_ID:TRAINING_SHA"
    )
    parser.add_argument(
        "--git-sha", required=True, help="Full reachable implementation commit SHA."
    )
    parser.add_argument(
        "--git-ref", help="Optional remote ref to fetch before exact detached checkout."
    )
    parser.add_argument(
        "--tracking-uri",
        required=True,
        help="HTTP(S) MLflow endpoint without credentials/query/fragment.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Fresh absolute directory beneath /mnt/projects.",
    )
    parser.add_argument("--experiment-name", default="MammaPrint concepts")
    parser.add_argument("--job-prefix", default="mammaprint-concepts")
    parser.add_argument("--username", default="kissmi")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument(
        "--k",
        action="extend",
        nargs="+",
        type=_positive,
        help="One or more component counts; repeatable (default: 16).",
    )
    parser.add_argument(
        "--gpu",
        choices=["H100", "A100", "A40", "mig-1g.10gb", "mig-2g.20gb"],
        default="A40",
    )
    parser.add_argument("--cpu", type=_positive, default=8)
    parser.add_argument("--memory", default="32Gi")
    parser.add_argument("--shm", default="16Gi")
    parser.add_argument("--ray-cpus", type=_positive, default=8)
    parser.add_argument(
        "--context",
        default=CONTEXT,
        help="Expected fixed private helper context; does not reroute jobs.",
    )
    parser.add_argument(
        "--namespace",
        default=NAMESPACE,
        help="Expected fixed helper namespace; does not reroute jobs.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly download and explain test after validation.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Create jobs (requires explicit cluster submission authorization).",
    )
    args = parser.parse_args(argv)
    try:
        for name in (
            "tracking_uri",
            "output_root",
            "experiment_name",
            "username",
            "image",
            "context",
            "namespace",
        ):
            _plain(getattr(args, name), name)
        if not re.fullmatch(r"[0-9a-fA-F]{40}", args.git_sha):
            raise ValueError("--git-sha must be an exact 40-character hexadecimal SHA")
        args.git_sha = args.git_sha.lower()
        if args.git_ref is not None and (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/_.-]*", args.git_ref)
            or ".." in args.git_ref
            or "//" in args.git_ref
            or "/." in args.git_ref
            or args.git_ref.endswith(("/", ".", ".lock"))
        ):
            raise ValueError("--git-ref must be a literal safe remote ref")
        sources = []
        seen = set()
        for value in args.source:
            parts = value.split(":")
            if (
                len(parts) != 2
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", parts[0])
                or not re.fullmatch(r"[0-9a-fA-F]{40}", parts[1])
            ):
                raise ValueError(
                    "--source requires safe RUN_ID:40-character-TRAINING_SHA"
                )
            if parts[0] in seen:
                raise ValueError(
                    "duplicate source run; each run must have exactly one training SHA"
                )
            seen.add(parts[0])
            sources.append((parts[0], parts[1].lower()))
        args.sources = sources
        args.k = args.k or [16]
        if len(set(args.k)) != len(args.k):
            raise ValueError("duplicate K would create duplicate source/K jobs")
        if args.ray_cpus > args.cpu:
            raise ValueError("--ray-cpus must not exceed --cpu")
        if _quantity_bytes(args.shm) > _quantity_bytes(args.memory):
            raise ValueError("--shm must not exceed --memory")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,28}[a-z0-9])?", args.job_prefix):
            raise ValueError(
                "--job-prefix must be a lowercase DNS label of at most 30 characters"
            )
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9_.-]{0,61}[a-z0-9])?", args.username):
            raise ValueError(
                "--username must be a valid lowercase Kubernetes label value"
            )
        if not args.image.startswith("cerit.io/") or any(
            character.isspace() for character in args.image
        ):
            raise ValueError("--image must be an approved cerit.io image reference")
        path = PurePosixPath(args.output_root)
        if (
            ".." in path.parts
            or not path.is_absolute()
            or path.parts[:3] != ("/", "mnt", "projects")
            or len(path.parts) <= 3
        ):
            raise ValueError(
                "--output-root must be an absolute subdirectory of /mnt/projects without '..'"
            )
        args.output_root = str(path)
        uri = urlsplit(args.tracking_uri)
        if (
            uri.scheme not in ("http", "https")
            or not uri.hostname
            or uri.username is not None
            or uri.password is not None
            or uri.query
            or uri.fragment
            or any(character.isspace() for character in args.tracking_uri)
        ):
            raise ValueError(
                "--tracking-uri must be HTTP(S), without credentials, whitespace, query, or fragment"
            )
        # Force malformed ports to fail in the offline validation stage too.
        _ = uri.port
    except ValueError as error:
        parser.error(str(error))
    return args


def _hydra(key: str, value: str) -> str:
    # Hydra only unescapes backslash runs before its matching quote (including
    # the closing quote). JSON escaping would double ordinary backslashes.
    quoted = re.sub(
        r'(\\*)"', lambda match: "\\" * (2 * len(match[1]) + 1) + '"', value
    )
    quoted = re.sub(r"(\\+)$", lambda match: match[0] * 2, quoted)
    return f'{key}="{quoted}"'


def _script(
    args: argparse.Namespace, run_id: str, training_sha: str, k: int, output: str
) -> list[str]:
    script = [
        "set -euo pipefail",
        "umask 077",
        shlex.join(["git", "clone", "--no-checkout", REPOSITORY, "workdir"]),
        "cd workdir",
    ]
    if args.git_ref:
        script.append(shlex.join(["git", "fetch", "origin", args.git_ref]))
    script.extend(
        [
            f"git cat-file -e {args.git_sha}^{{commit}} 2>/dev/null || git fetch origin {args.git_sha}",
            f"git checkout --detach {args.git_sha}",
            f'test "$(git rev-parse HEAD)" = {args.git_sha}',
            f"git cat-file -e {training_sha}^{{commit}} 2>/dev/null || git fetch origin {training_sha}",
            f"git cat-file -e {training_sha}^{{commit}}",
            "uv sync --frozen --extra explainability",
            shlex.join(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "--extra",
                    "explainability",
                    "python",
                    "-c",
                    "import torch, openslide, pyvips; "
                    "assert torch.cuda.is_available(), 'CUDA is required for native inference'; "
                    "print('CUDA', torch.version.cuda, torch.cuda.get_device_name(0)); "
                    "print('OpenSlide', openslide.__library_version__, 'libvips', pyvips.version(0))",
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
        "--extra",
        "explainability",
        "-m",
        "ml.concepts",
        "--config-name",
        "concepts_deployed",
        "action=fit_explain",
        _hydra("source_run_id", run_id),
        _hydra("training_sha", training_sha),
        _hydra("output_dir", output),
        _hydra("experiment_name", args.experiment_name),
        f"include_test={str(args.include_test).lower()}",
        "device=cuda",
        f"ray_cpus={args.ray_cpus}",
        f"fit.n_components={k}",
    ]
    script.append(shlex.join(command))
    return script


def build_plan(args: argparse.Namespace) -> dict:
    jobs = []
    for run_id, training_sha in args.sources:
        for k in args.k:
            output = f"{args.output_root}/{run_id}/k-{k}"
            identity = json.dumps([run_id, training_sha, args.git_sha, output, k])
            digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
            job_name = f"{args.job_prefix}-{run_id.lower()[:12].rstrip('-_').replace('_', '-')}-{digest}"
            jobs.append(
                {
                    "job_name": job_name,
                    "source_run_id": run_id,
                    "training_sha": training_sha,
                    "implementation_sha": args.git_sha,
                    "implementation_ref": args.git_ref,
                    "k": k,
                    "output_dir": output,
                    "script": _script(args, run_id, training_sha, k, output),
                }
            )
    return {
        "mode": "submit" if args.submit else "plan",
        "remote_verified": False,
        "repository": REPOSITORY,
        "image": args.image,
        "username": args.username,
        "target": {
            "context": args.context,
            "namespace": args.namespace,
            "helper_context": CONTEXT,
            "helper_namespace": NAMESPACE,
            "routing": "fixed private kube_jobs helper; flags are expectations",
        },
        "resources": {
            "gpu": args.gpu,
            "gpu_limit": 1,
            "cpu": args.cpu,
            "memory": args.memory,
            "shm": args.shm,
            "ray_cpus": args.ray_cpus,
        },
        "execution": {
            "profile": "concepts_deployed",
            "action": "fit_explain",
            "fit_split": "train",
            "explain_splits": ["val", "test"] if args.include_test else ["val"],
            "test_membership_audit": "metadata listing only; test bags downloaded only with --include-test",
            "cpu_stages": ["output metric", "semi-NMF fitting", "NNLS encoding"],
            "gpu_stage": "native PyTorch explanation inference",
            "supported_graph": "IdentityEncoder + MeanPool + single LinearHead; BCE/type or MSE/index; no attention term",
            "mlflow": "new destination run; source run read-only",
        },
        "unverified": [
            "remote Git reachability of both pins",
            "live source model/config compatibility",
            "cluster availability and GPU scheduling",
            "image CUDA/OpenSlide/libvips runtime",
            "resource sufficiency",
            "fresh output directories (core refuses nonempty results)",
        ],
        "jobs": jobs,
    }


def submit_plan(args: argparse.Namespace, plan: dict) -> list[dict]:
    # Imports and every kubeconfig/API side effect are behind --submit.
    helper = importlib.import_module("kube_jobs.submit_job")
    if (args.context, args.namespace) != (CONTEXT, NAMESPACE) or (
        helper.CLUSTER,
        helper.JOB_NAMESPACE,
    ) != (CONTEXT, NAMESPACE):
        raise RuntimeError(
            "target mismatch: installed private helper must use kubas-cluster/rationai-jobs-ns"
        )
    kubernetes = importlib.import_module("kubernetes")
    storage = importlib.import_module("kube_jobs.storage")
    # Fail closed if the named local context cannot be loaded; do not fall back
    # to the helper's in-cluster configuration path on a missing context.
    kubernetes.config.load_kube_config(context=args.context)
    endpoint = urlsplit(kubernetes.client.Configuration.get_default_copy().host)
    if (
        endpoint.scheme not in ("http", "https")
        or not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
    ):
        raise RuntimeError(
            "target kubeconfig server must be an HTTP(S) endpoint without credentials"
        )
    print(
        json.dumps(
            {
                "target_verified": {
                    "context": args.context,
                    "namespace": args.namespace,
                    "server": endpoint.geturl(),
                }
            }
        )
    )
    api = kubernetes.client.BatchV1Api()
    for job in plan["jobs"]:
        try:
            api.read_namespaced_job(name=job["job_name"], namespace=args.namespace)
        except kubernetes.client.rest.ApiException as error:
            if error.status != 404:
                raise
        else:
            raise RuntimeError(
                f"job {job['job_name']} already exists; choose a fresh output root/prefix"
            )
    confirmations = []
    for job in plan["jobs"]:
        helper.submit_job(
            job_name=job["job_name"],
            username=args.username,
            public=False,
            image=args.image,
            script=job["script"],
            cpu=args.cpu,
            memory=args.memory,
            shm=args.shm,
            gpu=args.gpu,
            storage=[storage.secure.DATA, storage.secure.PROJECTS],
        )
        try:
            created = api.read_namespaced_job(
                name=job["job_name"], namespace=args.namespace
            )
            if created.metadata.name != job["job_name"] or not created.metadata.uid:
                raise RuntimeError("missing job identity")
        except Exception as error:
            raise RuntimeError(
                f"submission not confirmed for {job['job_name']}; inspect before retrying"
            ) from error
        confirmation = {
            "job_name": job["job_name"],
            "uid": created.metadata.uid,
            "creation_confirmed": True,
            "execution_verified": False,
        }
        confirmations.append(confirmation)
        print(json.dumps(confirmation))
    return confirmations


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    print(json.dumps(plan, indent=2))
    if args.submit:
        submit_plan(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
