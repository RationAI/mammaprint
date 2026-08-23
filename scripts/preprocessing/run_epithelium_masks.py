"""Submit ONNX epithelium segmentation for mounted slides or image tiles."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from kube_jobs import storage, submit_job
from omegaconf import OmegaConf


DEFAULT_BRANCH = "codex/epithelium-onnx-job"
DEFAULT_REPOSITORY = "https://github.com/RationAI/MammaPrint.git"
DEFAULT_TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
DEFAULT_DATA_MAPPING = Path("/mnt/projects/mammaprint/data_mapping.csv")
DEFAULT_OUTPUT_DIR = Path("/mnt/projects/mammaprint/epithelium_onnx_masks")
DEFAULT_TILED_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/data/tiled/tissue_only/epithelium_512.yaml"
)
DEFAULT_MODEL_URI = (
    "mlflow-artifacts://mlflow.rationai-mlflow:5000/10/"
    "39f821ed5b964c71a603cc6db196f9fd/artifacts/"
    "checkpoints/epoch=19-step=32020/model.onnx/model.onnx"
)


def configured_tiled_dataset() -> str:
    config = OmegaConf.load(DEFAULT_TILED_CONFIG)
    return str(config.raw_uris.all)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit a job that downloads the epithelial ONNX model and "
            "generates masks for mounted slides or image tiles."
        )
    )
    parser.add_argument("--username", required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", dest="inputs", type=Path, nargs="+")
    inputs.add_argument(
        "--data-mapping",
        type=Path,
        help=f"CSV with a path column (default: {DEFAULT_DATA_MAPPING}).",
    )
    inputs.add_argument(
        "--tiled-dataset",
        help=(
            "MLflow artifact URI or local directory with slides/tiles parquet files "
            f"(default: raw_uris.all in {DEFAULT_TILED_CONFIG.name})."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kind", choices=("auto", "slide", "tile"), default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL_URI)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default="MammaPrint")
    parser.add_argument("--run-name", default="MammaPrint Epithelium ONNX Masks")
    parser.add_argument("--artifact-path", default="epithelium_masks")
    parser.add_argument("--source-mpp", type=float)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--provider",
        default="CUDAExecutionProvider",
        help=(
            "ONNX Runtime execution provider. GPU jobs require "
            "CUDAExecutionProvider and fail instead of falling back to CPU."
        ),
    )
    parser.add_argument("--job-name", default="mammaprint-epithelium-onnx")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--memory", default="32Gi")
    parser.add_argument("--gpu", default="A40", help="GPU type or 'none'.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def inference_command(args: argparse.Namespace) -> str:
    values = [
        "uv",
        "run",
        "python",
        "-m",
        "preprocessing.epithelium_onnx",
    ]
    if args.tiled_dataset:
        values.extend(("--tiled-dataset", args.tiled_dataset))
    elif args.inputs:
        values.extend(("--input", *(str(path) for path in args.inputs)))
    elif args.data_mapping:
        values.extend(("--data-mapping", str(args.data_mapping)))
    else:
        values.extend(("--tiled-dataset", configured_tiled_dataset()))
    values.extend(
        [
            "--output-dir",
            str(args.output_dir),
            "--kind",
            args.kind,
            "--model",
            args.model,
            "--tracking-uri",
            args.tracking_uri,
            "--experiment-name",
            args.experiment_name,
            "--run-name",
            args.run_name,
            "--artifact-path",
            args.artifact_path,
            "--batch-size",
            str(args.batch_size),
            "--provider",
            args.provider,
        ]
    )
    if args.source_mpp is not None:
        values.extend(("--source-mpp", str(args.source_mpp)))
    return shlex.join(values)


def job_commands(args: argparse.Namespace) -> list[str]:
    return [
        shlex.join(
            [
                "git",
                "clone",
                "--branch",
                args.branch,
                "--single-branch",
                args.repository,
                "workdir",
            ]
        ),
        "cd workdir",
        f"export MLFLOW_TRACKING_URI={shlex.quote(args.tracking_uri)}",
        "uv sync --frozen",
        inference_command(args),
    ]


def main() -> None:
    args = argument_parser().parse_args()
    commands = job_commands(args)
    if args.dry_run:
        print("\n".join(commands))
        return

    submit_job(
        job_name=args.job_name,
        username=args.username,
        image="cerit.io/rationai/base:2.0.6",
        cpu=args.cpu,
        memory=args.memory,
        gpu=None if args.gpu.lower() == "none" else args.gpu,
        public=False,
        script=commands,
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )


if __name__ == "__main__":
    main()
