# Copyright (c) The RationAI team.

import logging
import os
import random
import warnings

import git
import hydra
import lightning.pytorch
import lightning.pytorch.loggers
import mlflow
import omegaconf
import torch
from hydra.core.hydra_config import HydraConfig
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
)
from omegaconf import OmegaConf


# change logging level globally by hydra override:
# hydra.job_logging.root.level=DEBUG

log = logging.getLogger("task_cli")
torch.set_float32_matmul_precision("medium")

warnings.filterwarnings(action="ignore", message="Applied workaround for CuDNN issue,")
warnings.filterwarnings(action="ignore", message="Setuptools is replacing distutils.")


def seed_everything(seed, use_determinism=False):
    log.info(
        f"[Reproducibility] Seeding everything. Seed={seed}. Determinism={use_determinism}."
    )
    torch.manual_seed(seed)
    lightning.pytorch.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(use_determinism, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main(cfg: omegaconf.DictConfig, stage: str) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    seed_everything(cfg.seed, cfg.use_determinism)

    OmegaConf.save(
        config=cfg,
        f=HydraConfig.get().output_subdir + "/config_resolved.yaml",
        resolve=True,
    )

    task = hydra.utils.instantiate(
        cfg.task,
        stage=stage,
        _recursive_=True,
        _convert_="partial",
    )
    task.setup()
    task.run()
    task.teardown()


preset_hydra_main_decorator = hydra.main(
    version_base=None, config_path="../conf", config_name="default"
)

# Use Cache - True : all unseeded modules will be seeded with the same seed
OmegaConf.register_new_resolver(
    "random_seed", lambda: random.randint(0, 2**31), use_cache=True, replace=True
)


class Task:
    def __init__(
        self,
        ml: lightning.pytorch.LightningModule,
        datamodule: lightning.pytorch.LightningDataModule,
        trainer: lightning.pytorch.Trainer,
        stage: str,
        hyperparameters: dict | None = None,
    ):
        self.ml = ml
        self.datamodule = datamodule
        self.trainer = trainer
        self.hyperparameters = hyperparameters or {}
        if stage in ["fit", "test", "validate", "predict"]:
            self.stage = stage
            self.trainer_stage_fnc = getattr(self.trainer, stage)
        else:
            raise ValueError(f"Process {stage} not supported.")

        self.logger: lightning.pytorch.loggers.MLFlowLogger = self.trainer.logger
        self.client: mlflow.MlflowClient = self.logger.experiment
        self.run_id = self.logger.run_id
        experiment_id = self.logger.experiment_id
        mlflow.start_run(self.run_id, experiment_id=experiment_id)

    def run(self):
        self.trainer_stage_fnc(self.ml, datamodule=self.datamodule)

    def setup(self) -> None:
        self.client.log_artifacts(
            run_id=self.run_id,
            local_dir=HydraConfig.get().output_subdir,
            artifact_path="conf",
        )
        self.client.set_tag(run_id=self.run_id, key="stage", value=self.stage)
        self.logger.log_hyperparams(self.hyperparameters)
        self.__log_git_status()

    def teardown(self) -> None:
        """Saves the log file to MLFlow."""
        hydra_conf = HydraConfig.get()
        log_fp = hydra_conf.job_logging.handlers.file.filename
        self.client.log_artifact(
            run_id=self.run_id, local_path=log_fp, artifact_path="logs"
        )

    def __log_git_status(self) -> None:
        """MLFlow claims to do this automatically, but it seems that it only resolves the git commit hash.

        This function resolves the git repo url and branch as well.
        """
        path_to_git_repo = hydra.utils.get_original_cwd()
        try:
            repo = git.Repo(path_to_git_repo)
        except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError) as err:
            log.warning(f"Cannot get git repo: {err}")
            return

        try:
            if not repo.remotes:  # if empty
                raise StopIteration  # xD log.warning("Cannot get git remote url")

            remote_url = next(repo.remotes[0].urls)
            self.client.set_tag(
                run_id=self.run_id, key=MLFLOW_GIT_REPO_URL, value=remote_url
            )
            # MLflow's UI does not show this; we add custom tag
            self.client.set_tag(
                run_id=self.run_id, key="git.repo_url", value=remote_url
            )
        except StopIteration:
            log.warning("Cannot get git remote url")

        if not repo.head.is_detached:
            branch = repo.active_branch.name
            self.client.set_tag(run_id=self.run_id, key=MLFLOW_GIT_BRANCH, value=branch)
            # MLFlow's UI does not show this; we add custom tag:
            self.client.set_tag(run_id=self.run_id, key="git.branch", value=branch)
        else:
            log.warning("Cannot get git branch ('detached HEAD' state)")

        commit_hash = repo.head.commit.hexsha
        self.client.set_tag(
            run_id=self.run_id, key=MLFLOW_GIT_COMMIT, value=commit_hash
        )
