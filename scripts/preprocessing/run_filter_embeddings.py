"""Submit one CPU-only embedding-filtering job."""

from __future__ import annotations

import argparse
import hashlib
import math
import shlex

from kube_jobs import storage, submit_job


USERNAME = "kissmi"
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
DEFAULT_STRATEGIES = ("epithelium_only", "or", "and")
STRATEGY_CHOICES = (*DEFAULT_STRATEGIES, "qc_only")
QC_MASKS = ("blur", "folding", "residual")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level", type=int, choices=sorted(TISSUE_EMBEDDING_URIS), required=True
    )
    parser.add_argument("--tiling-uri", required=True)
    parser.add_argument("--embeddings-uri")
    parser.add_argument("--embeddings-path")
    parser.add_argument("--epithelium-threshold", type=float, default=0.25)
    parser.add_argument("--cancer-threshold", type=float, default=0.5)
    for name in QC_MASKS:
        parser.add_argument(f"--{name}-threshold", type=float)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGY_CHOICES,
        default=list(DEFAULT_STRATEGIES),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--upload-max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--git-ref",
        default=GIT_BRANCH,
        help="Published branch or commit containing the filtering code.",
    )
    return parser.parse_args()


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "")


def main() -> None:
    args = _parse_args()
    if args.upload_only and not args.output_dir:
        raise ValueError("--upload-only requires --output-dir")
    if args.no_upload and (not args.output_dir or args.upload_only):
        raise ValueError(
            "--no-upload requires --output-dir and cannot use --upload-only"
        )
    qc = {
        name: getattr(args, f"{name}_threshold")
        for name in QC_MASKS
        if getattr(args, f"{name}_threshold") is not None
    }
    for name, value in qc.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} threshold must be finite and in [0, 1]")
    if "qc_only" in args.strategies and not qc:
        raise ValueError("qc_only requires at least one QC threshold")
    embeddings_uri = args.embeddings_uri or TISSUE_EMBEDDING_URIS[args.level]
    # An explicit source URI must not silently use the default tissue embedding directory.
    embeddings_path = args.embeddings_path
    if embeddings_path is None and args.embeddings_uri is None:
        embeddings_path = TISSUE_EMBEDDING_PATHS[args.level]
    strategies = tuple(dict.fromkeys(args.strategies))
    strategy_token = "-".join(
        {"epithelium_only": "epi", "or": "or", "and": "and", "qc_only": "qc"}[strategy]
        for strategy in strategies
    )
    job_name = (
        f"mammaprint-filter-embeddings-l{args.level}-"
        f"{strategy_token}-epi{_token(args.epithelium_threshold)}-"
        f"c{_token(args.cancer_threshold)}"
    )
    if qc:
        suffix = "-".join(
            f"{name}{str(float(value)).replace('.', '')}" for name, value in qc.items()
        )
        job_name += f"-{suffix}"
    # Kubernetes job names must fit within 63 characters, including strict-QC settings.
    if len(job_name) > 63:
        digest = hashlib.sha256(job_name.encode()).hexdigest()[:10]
        job_name = f"{job_name[:52].rstrip('-')}-{digest}"
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
        "--epithelium-threshold",
        f"{args.epithelium_threshold:g}",
        "--cancer-threshold",
        f"{args.cancer_threshold:g}",
        "--strategies",
        *strategies,
        "--upload-max-retries",
        str(args.upload_max_retries),
    ]
    if embeddings_path:
        command.extend(["--embeddings-path", embeddings_path])
    for name, value in qc.items():
        command.extend([f"--{name}-threshold", str(value)])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.upload_only:
        command.append("--upload-only")
    if args.no_upload:
        command.append("--no-upload")
    filter_command = shlex.join(command)

    if args.dry_run:
        print(f"Job: {job_name}")
        print(f"Command: {filter_command}")
        print(f"Git ref: {args.git_ref}")
        return

    submit_job(
        job_name=job_name,
        username=USERNAME,
        image="cerit.io/rationai/base:2.0.6",
        cpu=16,
        memory="64Gi",
        shm="16Gi",
        gpu=None,
        public=False,
        script=[
            "git clone https://github.com/rationAI/mammaprint workdir",
            "cd workdir",
            f"git checkout {shlex.quote(args.git_ref)}",
            "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
            "uv sync --frozen",
            filter_command,
        ],
        storage=[storage.secure.PROJECTS],
    )


if __name__ == "__main__":
    main()
