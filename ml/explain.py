"""Generate post-hoc tile explanations from a trained embedding-MIL checkpoint.

The source checkpoint's MLflow URI identifies both the immutable weights and the
training run whose logged Hydra overrides reconstruct the exact model/data graph.
This command creates a separate MLflow run and never resumes or mutates training.

Example:
    uv run -m ml.explain \
      --checkpoint-uri \
      mlflow-artifacts:/3/<run-id>/artifacts/checkpoints/best \
      --split test --override explain.ig.steps=32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit, urlunsplit

import hydra
import mlflow
import pandas as pd
from mlflow import MlflowClient
from omegaconf import OmegaConf

from ml.explainability.checkpoint import (
    CheckpointBootstrap,
    bootstrap_checkpoint,
    compose_training_config,
    load_lightning_checkpoint,
    stable_config_hash,
    validate_replayed_source_config,
)
from ml.explainability.runner import run_cohort, validate_module
from ml.models.module import MammaprintModule


if TYPE_CHECKING:
    from collections.abc import Sequence


log = logging.getLogger(__name__)
DEFAULT_XOPAT_UI_URI = "https://xopat.rationai.cloud.trusted.e-infra.cz/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-uri",
        required=True,
        help="MLflow checkpoint.ckpt/checkpoints/best URI, or a local .ckpt path.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Configured embedding split to explain (default: test).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra override applied after the recovered training overrides; repeatable.",
    )
    parser.add_argument(
        "--slide-id",
        action="append",
        default=[],
        help="Restrict inference to a slide ID; repeatable (default: complete split).",
    )
    parser.add_argument(
        "--run-name",
        help="Name for the new explanation MLflow run.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI).",
    )
    parser.add_argument(
        "--mlflow-ui-uri",
        default=os.environ.get("MLFLOW_UI_URI"),
        help=(
            "Browser-facing MLflow base URI used only to print the pathologist "
            "review link (defaults to MLFLOW_UI_URI)."
        ),
    )
    parser.add_argument(
        "--xopat-ui-uri",
        default=os.environ.get("XOPAT_UI_URI", DEFAULT_XOPAT_UI_URI),
        help=(
            "Browser-facing xOpat base URI made available to generated reports "
            f"(default: XOPAT_UI_URI or {DEFAULT_XOPAT_UI_URI})."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cpu, cuda, cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve checkpoint/config and validate the model without reading slides.",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _effective_caller_overrides(args: argparse.Namespace) -> list[str]:
    return [*args.override, f"explain.split={args.split}"]


def _instantiate_module(config: Any, bootstrap: CheckpointBootstrap) -> tuple[Any, Any]:
    module = hydra.utils.instantiate(config.ml)
    if not isinstance(module, MammaprintModule):
        raise TypeError(
            "Recovered ml config did not instantiate MammaprintModule; got "
            f"{type(module).__module__}.{type(module).__name__}."
        )
    load_info = load_lightning_checkpoint(module, bootstrap.checkpoint_path)
    validate_module(module, config)
    return module, load_info


def _source_run(
    client: MlflowClient,
    bootstrap: CheckpointBootstrap,
) -> Any | None:
    if bootstrap.source_run_id is None:
        return None
    run = client.get_run(bootstrap.source_run_id)
    expected_experiment = bootstrap.checkpoint.source_experiment_id
    if expected_experiment and run.info.experiment_id != expected_experiment:
        raise RuntimeError(
            "Checkpoint URI experiment ID does not match the source run: "
            f"URI={expected_experiment}, run={run.info.experiment_id}."
        )
    return run


def _experiment_id(client: MlflowClient, config: Any, source_run: Any | None) -> str:
    if source_run is not None:
        return str(source_run.info.experiment_id)
    experiment_name = str(config.metadata.experiment_name)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return str(experiment.experiment_id)
    return str(client.create_experiment(experiment_name))


def _explanation_run_name(
    requested: str | None,
    bootstrap: CheckpointBootstrap,
    source_run: Any | None,
) -> str:
    if requested:
        return requested
    if source_run is not None:
        source_name = source_run.data.tags.get("mlflow.runName")
        if source_name:
            return f"🗣️ Explaining: {source_name}"
    identity = bootstrap.source_run_id or bootstrap.source_hash[:12]
    return f"🗣️ Explaining: MammaPrint - {identity}"


def _mlflow_run_url(
    ui_uri: str | None,
    experiment_id: str,
    run_id: str,
) -> str | None:
    """Build the stable MLflow UI route without using the tracking backend URI."""
    if ui_uri is None or not ui_uri.strip():
        return None
    parsed = urlsplit(ui_uri.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--mlflow-ui-uri must be an absolute HTTP(S) URL.")
    base_path = parsed.path.rstrip("/")
    base = urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))
    return (
        f"{base}/#/experiments/{quote(str(experiment_id), safe='')}"
        f"/runs/{quote(str(run_id), safe='')}"
    )


def _artifact_path(parent: str, filename: str) -> str:
    parent = parent.strip("/")
    return f"{parent}/{filename}" if parent else filename


def _log_slide_key_table(slide_keys_path: Path, artifact_path: str) -> str:
    """Log the CSV key as an MLflow table that renders directly in the UI."""
    artifact_file = _artifact_path(artifact_path, "slide_keys.json")
    table = pd.read_csv(slide_keys_path, dtype={"record_num": "string"})
    mlflow.log_table(table, artifact_file=artifact_file)
    return artifact_file


def _dry_run_document(
    config: Any,
    module: MammaprintModule,
    bootstrap: CheckpointBootstrap,
    load_info: Any,
    slide_ids: list[str],
    resolved_config_hash: str,
    source_critical_config_hash: str | None,
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "provenance": bootstrap.provenance(),
        "aggregator": type(module.aggregator).__name__,
        "head": type(module.head).__name__,
        "head_out_dim": module.head.out_dim,
        "label_mode": str(config.label_mode),
        "split": str(config.explain.split),
        "slide_filter": slide_ids,
        "resolved_config_hash": resolved_config_hash,
        "source_critical_config_hash": source_critical_config_hash,
        "load_info": {
            "state_dict_keys": load_info.state_dict_keys,
            "epoch": load_info.epoch,
            "global_step": load_info.global_step,
        },
        "explain": OmegaConf.to_container(config.explain, resolve=True),
    }


def _save_configs(
    config: Any,
    bootstrap: CheckpointBootstrap,
    output_dir: Path,
) -> None:
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_dir / "config-resolved.yaml", resolve=True)
    if bootstrap.source_config_path is not None:
        shutil.copy2(
            bootstrap.source_config_path,
            config_dir / "source-config-resolved.yaml",
        )
    if bootstrap.hydra_config_path is not None:
        shutil.copy2(
            bootstrap.hydra_config_path,
            config_dir / "source-hydra.yaml",
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging()
    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    with tempfile.TemporaryDirectory(prefix="mammaprint-explain-") as temporary:
        temporary_root = Path(temporary)
        bootstrap = bootstrap_checkpoint(
            args.checkpoint_uri,
            temporary_root / "bootstrap",
            caller_overrides=_effective_caller_overrides(args),
            tracking_uri=tracking_uri,
            client=client,
        )
        config = compose_training_config(
            bootstrap.effective_overrides,
            config_name="explain",
        )
        source_critical_config_hash: str | None = None
        if bootstrap.source_config_path is not None:
            source_config = OmegaConf.load(bootstrap.source_config_path)
            source_seed = OmegaConf.select(source_config, "seed", default=None)
            if not isinstance(source_seed, int):
                raise ValueError(
                    "The source resolved config does not contain an integer seed."
                )
            # The training seed may originate from a random Hydra resolver and thus
            # be absent from task overrides. Restore its logged resolved value so
            # the effective config and its hash are repeatable.
            config.seed = source_seed
            source_critical_config_hash = validate_replayed_source_config(
                config,
                bootstrap.source_config_path,
            )
        resolved_config_hash = stable_config_hash(
            OmegaConf.to_container(config, resolve=True)
        )
        module, load_info = _instantiate_module(config, bootstrap)
        source_run = _source_run(client, bootstrap)

        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_document(
                        config,
                        module,
                        bootstrap,
                        load_info,
                        list(args.slide_id),
                        resolved_config_hash,
                        source_critical_config_hash,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        experiment_id = _experiment_id(client, config, source_run)
        run_name = _explanation_run_name(args.run_name, bootstrap, source_run)
        output_dir = temporary_root / "output"
        output_dir.mkdir(parents=True)
        _save_configs(config, bootstrap, output_dir)
        provenance = {
            **bootstrap.provenance(),
            "override_hash": bootstrap.override_hash,
            "config_hash": resolved_config_hash,
            "source_critical_config_hash": source_critical_config_hash,
            "checkpoint_load": {
                "state_dict_keys": load_info.state_dict_keys,
                "epoch": load_info.epoch,
                "global_step": load_info.global_step,
            },
        }
        tags = {
            "job_type": "tile_explainability",
            "mlflow.note.content": (
                "Pathologist review of signed tile explanations, predictions, "
                "slide keys, and model-faithfulness checks. Open Artifacts → "
                "explanations → report.html."
            ),
            "report.audience": "pathologist",
            "report.kind": "tile_explainability",
            "report.status": "generating",
            "source_training_run_id": bootstrap.source_run_id or "local",
            "source_checkpoint_uri": str(args.checkpoint_uri),
            "source_hash": bootstrap.source_hash,
            "config_hash": resolved_config_hash,
            "aggregator": type(module.aggregator).__name__,
            "head": type(module.head).__name__,
            "split": str(config.explain.split),
        }

        result = None
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
        ) as explanation_run:
            review_url = _mlflow_run_url(
                args.mlflow_ui_uri,
                experiment_id,
                explanation_run.info.run_id,
            )
            if review_url is not None:
                mlflow.set_tag("pathologist_review_url", review_url)
                print(f"PATHOLOGIST_REVIEW_URL={review_url}", flush=True)
            log.info(
                "Starting explanation run %s from checkpoint source %s.",
                explanation_run.info.run_id,
                bootstrap.source_run_id or "local",
            )
            provenance.update(
                {
                    "explanation_run_id": explanation_run.info.run_id,
                    "explanation_experiment_id": str(
                        explanation_run.info.experiment_id
                    ),
                    "mlflow_run_url": review_url,
                    "mlflow_ui_uri": args.mlflow_ui_uri,
                    "xopat_ui_uri": args.xopat_ui_uri,
                }
            )
            mlflow.log_params(
                {
                    "checkpoint_sha256": bootstrap.checkpoint_sha256,
                    "config_hash": resolved_config_hash,
                    "override_hash": bootstrap.override_hash,
                    "source_hash": bootstrap.source_hash,
                    "split": str(config.explain.split),
                    "aggregator": type(module.aggregator).__name__,
                    "head": type(module.head).__name__,
                    "ig_steps": int(config.explain.ig.steps),
                }
            )
            try:
                result = run_cohort(
                    config=config,
                    module=module,
                    output_dir=output_dir,
                    provenance=provenance,
                    slide_ids=list(args.slide_id),
                    device=args.device,
                )
                mlflow.log_metrics(
                    {
                        "explanations/successful_slides": result.successful_slides,
                        "explanations/failed_slides": result.failed_slides,
                        **{
                            f"explanations/{name}": value
                            for name, value in result.metrics.items()
                        },
                    }
                )
                if result.successful_slides == 0:
                    raise RuntimeError("No slide explanations completed successfully.")
                slide_key_table = _log_slide_key_table(
                    result.slide_keys_path,
                    str(config.explain.output.artifact_path),
                )
            except Exception:
                mlflow.set_tag("report.status", "failed")
                if any(output_dir.iterdir()):
                    try:
                        mlflow.log_artifacts(
                            str(output_dir),
                            artifact_path=str(config.explain.output.artifact_path),
                        )
                    except Exception:
                        log.exception("Could not upload partial explanation artifacts.")
                raise
            else:
                if any(output_dir.iterdir()):
                    try:
                        mlflow.log_artifacts(
                            str(output_dir),
                            artifact_path=str(config.explain.output.artifact_path),
                        )
                    except Exception:
                        mlflow.set_tag("report.status", "failed")
                        raise
                mlflow.set_tags(
                    {
                        "report.status": "ready",
                        "report.artifact": _artifact_path(
                            str(config.explain.output.artifact_path),
                            result.report_path.name,
                        ),
                        "report.slide_key_table": slide_key_table,
                        "report.review_sheet": _artifact_path(
                            str(config.explain.output.artifact_path),
                            result.pathologist_review_path.name,
                        ),
                        "report.successful_slides": result.successful_slides,
                        "report.failed_slides": result.failed_slides,
                    }
                )

            if result is None:
                raise RuntimeError("No slide explanations completed successfully.")
            log.info(
                "Completed %d slides (%d failed); artifacts logged to run %s.",
                result.successful_slides,
                result.failed_slides,
                explanation_run.info.run_id,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
