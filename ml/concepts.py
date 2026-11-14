"""Replicate the laboratory IBD concept workflow for frozen MammaPrint models."""

import contextlib
import hashlib
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ml.concept_explainability.data import audit_splits
from ml.concept_explainability.evaluation import (
    evaluate_concepts,
    fit_dictionary,
    save_json,
)
from ml.concept_explainability.model import (
    REPOSITORY,
    SourceRun,
    file_sha256,
    load_source_run,
    verify_training_revision,
)
from ml.concept_explainability.report import build_report


def run(config: DictConfig) -> Path:
    """Fit once per corpus, or explain a checkpoint using that fixed dictionary."""
    action = str(config.action)
    if config.log_to_mlflow and os.environ.get("MLFLOW_RUN_ID"):
        raise RuntimeError(
            "Inherited MLFLOW_RUN_ID could resume a training run; unset it before logging explanations."
        )
    if action not in {"fit", "explain"}:
        raise ValueError("action must be fit or explain")
    if action == "fit" and config.split != "train":
        raise ValueError("Dictionaries and centering may only be fitted on train.")
    output = Path(str(config.output_dir)).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "output_dir must be empty; previous results are not overwritten."
        )
    if config.local_source_config:
        source_config = OmegaConf.load(config.local_source_config)
        OmegaConf.to_container(source_config, resolve=True, throw_on_missing=True)
        verify_training_revision(config.training_sha, config=source_config)
        checkpoint = Path(config.local_checkpoint) if config.local_checkpoint else None
        source = SourceRun(
            str(config.source_run_id),
            config.training_sha,
            source_config,
            checkpoint,
            file_sha256(checkpoint) if checkpoint else None,
        )
    else:
        source = load_source_run(
            str(config.source_run_id),
            output.parent / ".source-cache",
            download_weights=action == "explain",
            training_sha=config.training_sha,
        )
    split_audit = audit_splits(source)
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(config.device)
    if device not in {"cpu", "cuda"} or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("device must be cpu, auto or an available cuda device")
    mlflow = None
    publication = None
    context = contextlib.nullcontext()
    if config.log_to_mlflow:
        import mlflow

        if mlflow.active_run() is not None:
            raise RuntimeError(
                "An MLflow run is already active; refusing to reuse a training run."
            )
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(str(config.experiment_name))
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(str(config.experiment_name))
        )
        context = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"MammaPrint concepts: {action}",
            tags={
                "source_run_id": source.run_id,
                "source_training_sha": source.training_sha,
                "fixture_only": str(bool(config.fixture_mode)),
            },
        )
        publication = {
            "run_id": context.info.run_id,
            "experiment_id": str(experiment_id),
        }
    with context:
        if action == "fit":
            if config.fixture_mode:
                fit_dictionary(source, config.fit, output, use_ray=False)
            else:
                import ray

                with ray.init(
                    num_cpus=int(config.ray_cpus),
                    runtime_env={"excludes": [".git", ".venv"]},
                ):
                    fit_dictionary(source, config.fit, output)
        else:
            if not config.dictionary_dir:
                raise ValueError(
                    "explain requires dictionary_dir from an existing training fit."
                )
            dictionary = Path(config.dictionary_dir).expanduser().resolve()
            dictionary_manifest = json.loads((dictionary / "manifest.json").read_text())
            if dictionary_manifest["fixture_only"] and not config.fixture_mode:
                raise ValueError(
                    "A fixture dictionary cannot be used for a full scientific report."
                )
            evaluate_concepts(
                source,
                dictionary,
                str(config.split),
                output,
                device=device,
                step_sds=float(config.step_sds),
            )
            build_report(source, dictionary, output, output, publication=publication)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        implementation_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip()
        manifest.update(
            {
                "implementation_sha": implementation_sha,
                "source_config_sha256": hashlib.sha256(
                    OmegaConf.to_yaml(source.config, resolve=True).encode()
                ).hexdigest(),
                "source_origin": "local_explicit"
                if config.local_source_config
                else "mlflow",
                "split_audit": split_audit,
                "publication": publication,
                "dependencies": {
                    name: version(name)
                    for name in ("torch", "numpy", "pandas", "ray", "scikit-learn")
                },
            }
        )
        save_json(manifest_path, manifest)
        # Source configs may include credentials; log only this workflow's settings.
        if mlflow is not None:
            mlflow.log_artifacts(str(output))
    print(f"Concept {action} output: {output}")
    return output


@hydra.main(config_path="../configs", config_name="concepts", version_base=None)
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
