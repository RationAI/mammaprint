# Standard Imports
import logging
import warnings
from abc import ABC, abstractmethod

import git
import hydra
import lightning.pytorch
import lightning.pytorch.loggers
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
)


# change logging level globally by hydra override:
# hydra.job_logging.root.level=DEBUG

log = logging.getLogger("task_cli")
torch.set_float32_matmul_precision("medium")

warnings.filterwarnings(action="ignore", message="Applied workaround for CuDNN issue,")
warnings.filterwarnings(action="ignore", message="Setuptools is replacing distutils.")


class AbstractTask(ABC):
    client: str | None = None
    run_id: str | None = None
    stage: str = "run"

    @abstractmethod
    def run(self):
        """Defines the main functionality of the task."""

    def setup(self):
        if self.client is None:
            self.client = mlflow.MlflowClient()
        if self.run_id is None:
            self.run_id = mlflow.active_run().info.run_id
        self.client.log_artifacts(
            run_id=self.run_id,
            local_dir=HydraConfig.get().output_subdir,
            artifact_path="conf",
        )
        self._log_git_status()
        self.client.set_tag(run_id=self.run_id, key="stage", value=self.stage)
        self._log_git_status()

    def teardown(self) -> None:
        """Saves the log file to MLflow."""
        hydra_conf = HydraConfig.get()
        log_fp = hydra_conf.job_logging.handlers.file.filename
        self.client.log_artifact(
            run_id=self.run_id, local_path=log_fp, artifact_path="logs"
        )

    def _log_git_status(self) -> None:
        """This function resolves the git repo url and branch as well.

        MLfow claims to do this automatically, but it seems that it only resolves the git commit hash.
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
            # MLflow's UI does not show this; we add custom tag:
            self.client.set_tag(run_id=self.run_id, key="git.branch", value=branch)
        else:
            log.warning("Cannot get git branch ('detached HEAD' state)")

        commit_hash = repo.head.commit.hexsha
        self.client.set_tag(
            run_id=self.run_id, key=MLFLOW_GIT_COMMIT, value=commit_hash
        )


class Task(AbstractTask):
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

    def teardown(self):
        super().teardown()
        self.logger.log_hyperparams(self.hyperparameters)
