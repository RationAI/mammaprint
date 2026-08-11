"""Submit an embedding-to-mask explanation job on the training cluster.

The submitted pod downloads the immutable source checkpoint/configuration and starts
a *new* MLflow run through :mod:`ml.explain`.  Use ``--dry-run`` to inspect the full
quoted job specification without contacting Kubernetes.

Example:
    uv run python scripts/ml/submit_explain.py \
        --checkpoint-uri \
        mlflow-artifacts:/3/<run-id>/artifacts/checkpoints/best \
        --split test --override explain.ig.steps=32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence


USER = "kissmi"
MLFLOW_URI = "http://mlflow-s3.rationai-mlflow"
MLFLOW_UI_URI = "https://mlflow.rationai.cloud.trusted.e-infra.cz/"
XOPAT_UI_URI = "https://xopat.rationai.cloud.trusted.e-infra.cz/"
DEFAULT_BRANCH = "feat/explainability"
IMAGE = "cerit.io/rationai/base:2.0.6"
REPOSITORY = "https://github.com/rationAI/mammaprint"
CPU = 16
MEMORY = "48Gi"
STORAGE_NAMES = ("secure.DATA", "secure.PROJECTS")


@dataclass(frozen=True)
class ExplainJobSpec:
    """Serializable job shape used for both dry-run output and submission."""

    job_name: str
    username: str
    image: str
    cpu: int
    memory: str
    gpu: str
    public: bool
    script: tuple[str, ...]
    storage: tuple[str, ...] = STORAGE_NAMES

    def printable(self) -> str:
        """Stable JSON representation suitable for review and shell logs."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _quoted_command(arguments: Sequence[str]) -> str:
    return " ".join(shlex.quote(argument) for argument in arguments)


def _validate_branch(branch: str) -> str:
    # It becomes a positional git argument after shell quoting.  Reject option-like
    # or revision-expression inputs as an extra defence beyond shlex.quote.
    if (
        not branch
        or branch.startswith("-")
        or ".." in branch
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch)
    ):
        raise ValueError(f"Unsafe git branch name: {branch!r}")
    return branch


def kubernetes_job_name(checkpoint_uri: str, run_name: str | None) -> str:
    """Build a deterministic RFC-1123 job name no longer than 63 characters."""
    digest = hashlib.sha256(checkpoint_uri.encode("utf-8")).hexdigest()[:8]
    identity = run_name or digest
    slug = re.sub(r"[^a-z0-9]+", "-", identity.lower()).strip("-")
    if not slug:
        slug = digest
    suffix = f"-{digest}"
    prefix = "mammaprint-explain-"
    available = 63 - len(prefix) - len(suffix)
    slug = slug[:available].rstrip("-") or digest
    return f"{prefix}{slug}{suffix}"


def build_job_spec(args: argparse.Namespace) -> ExplainJobSpec:
    """Translate parsed CLI arguments into a safely quoted cluster job."""
    branch = _validate_branch(args.branch)
    explain_arguments = [
        "uv",
        "run",
        "-m",
        "ml.explain",
        "--checkpoint-uri",
        args.checkpoint_uri,
        "--split",
        args.split,
        "--tracking-uri",
        args.tracking_uri,
        "--mlflow-ui-uri",
        args.mlflow_ui_uri,
        "--xopat-ui-uri",
        args.xopat_ui_uri,
    ]
    if args.run_name:
        explain_arguments.extend(("--run-name", args.run_name))
    for slide_id in args.slide_id:
        explain_arguments.extend(("--slide-id", slide_id))
    for override in args.override:
        explain_arguments.extend(("--override", override))

    script = (
        _quoted_command(("git", "clone", REPOSITORY, "workdir")),
        _quoted_command(("cd", "workdir")),
        _quoted_command(("git", "checkout", "--quiet", branch)),
        f"export MLFLOW_TRACKING_URI={shlex.quote(args.tracking_uri)}",
        f"export MLFLOW_UI_URI={shlex.quote(args.mlflow_ui_uri)}",
        f"export XOPAT_UI_URI={shlex.quote(args.xopat_ui_uri)}",
        _quoted_command(("uv", "sync", "--frozen")),
        _quoted_command(explain_arguments),
    )
    return ExplainJobSpec(
        job_name=kubernetes_job_name(args.checkpoint_uri, args.run_name),
        username=USER,
        image=IMAGE,
        cpu=CPU,
        memory=MEMORY,
        gpu=args.gpu,
        public=False,
        script=script,
    )


def submit(spec: ExplainJobSpec) -> None:
    """Submit a reviewed specification with the repository's secure data mounts."""
    from kube_jobs import storage, submit_job

    kwargs: dict[str, Any] = {
        "job_name": spec.job_name,
        "username": spec.username,
        "image": spec.image,
        "cpu": spec.cpu,
        "memory": spec.memory,
        "gpu": spec.gpu,
        "public": spec.public,
        "script": list(spec.script),
        "storage": [storage.secure.DATA, storage.secure.PROJECTS],
    }
    submit_job(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-uri",
        required=True,
        help="Local checkpoint path or MLflow checkpoint.ckpt/checkpoints/best URI.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to explain (default: test).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Caller Hydra override applied after recovered training overrides; repeatable.",
    )
    parser.add_argument(
        "--slide-id",
        action="append",
        default=[],
        help="Restrict inference to a slide id; repeatable (default: complete split).",
    )
    parser.add_argument(
        "--run-name",
        help="Name for the new explanation MLflow run (never the source training run).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=MLFLOW_URI,
        help=f"MLflow tracking URI (default: {MLFLOW_URI}).",
    )
    parser.add_argument(
        "--mlflow-ui-uri",
        default=os.environ.get("MLFLOW_UI_URI", MLFLOW_UI_URI),
        help=(
            "Browser-facing MLflow base URI for the pathologist review link "
            f"(default: MLFLOW_UI_URI or {MLFLOW_UI_URI})."
        ),
    )
    parser.add_argument(
        "--xopat-ui-uri",
        default=os.environ.get("XOPAT_UI_URI", XOPAT_UI_URI),
        help=(
            "Browser-facing xOpat base URI for generated slide-review links "
            f"(default: XOPAT_UI_URI or {XOPAT_UI_URI})."
        ),
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"Git branch containing the explainer (default: {DEFAULT_BRANCH}).",
    )
    parser.add_argument("--gpu", default="A40", help="GPU type (default: A40).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact job specification without submitting it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = build_job_spec(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.dry_run:
        print(spec.printable())
        return 0

    submit(spec)
    print(f"Submitted explanation job: {spec.job_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
