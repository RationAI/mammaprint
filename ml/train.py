"""Entrypoint for MammaPrint MIL models — train / validate / test / predict.

Mirrors the preprocessing entrypoints: Hydra composes the config, ``@autolog``
sets up MLflow logging, and every object (datamodule, module, trainer) is built
with :func:`hydra.utils.instantiate`. The Lightning stage is selected by
``config.mode`` (``fit``/``validate``/``test``/``predict``), and ``config.checkpoint``
optionally resumes / loads weights.

Run with, e.g.::

    uv run -m ml.train +experiment=ml/train_mil_embeddings \
        data/embedded=demo_l3_embed
    uv run -m ml.train +experiment=ml/train_mil_embeddings \
        data/embedded=demo_l3_embed mode=test checkpoint=/path/best.ckpt
"""

import logging
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import hydra
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog

from ml.loggers import MLFlowLogger

from ml.logging.pyramid_audit import ArtifactLogger, log_pyramid_audit


log = logging.getLogger(__name__)


# Resolves ${random_seed:} in the config; cached so all consumers share one seed.
OmegaConf.register_new_resolver(
    "random_seed", lambda: random.randint(0, 2**31), use_cache=True, replace=True
)


class _AuditDataModule(Protocol):
    def setup(self, stage: str | None = None) -> None: ...

    def get_dataset(self, split: str) -> object: ...


def _log_pyramid_audits(
    datamodule: _AuditDataModule,
    logger: ArtifactLogger,
    mode: str,
) -> None:
    """Upload audits for the dataset splits consumed by this Lightning stage."""
    datamodule.setup(mode)
    splits = {
        "fit": ("train", "val"),
        "validate": ("val",),
        "test": ("test",),
        "predict": ("test",),
    }.get(mode, ())

    with TemporaryDirectory(prefix="mammaprint-pyramid-audit-") as temp_dir:
        output_root = Path(temp_dir)
        for split in splits:
            dataset = datamodule.get_dataset(split)
            log_pyramid_audit(
                dataset,
                logger,
                output_root / split,
                artifact_path=f"pyramid_selection/{split}",
            )


@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    seed_everything(int(config.seed), workers=True)

    # _recursive_=False: keep the per-split dataset nodes as DictConfig so the
    # DataModule can instantiate them lazily per stage in setup().
    datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)
    module = hydra.utils.instantiate(config.ml)
    trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Build only the current stage's splits. Multiscale datasets expose audit
    # tables; single-scale datasets make this a no-op at the logging boundary.
    try:
        _log_pyramid_audits(datamodule, logger, str(config.mode))
    except Exception:
        logger.finalize(status="failed")
        raise

    # config.mode selects the Lightning stage (fit/validate/test/predict);
    # config.checkpoint (or null) resumes training / loads weights for eval.
    run_stage = getattr(trainer, config.mode)
    run_stage(module, datamodule=datamodule, ckpt_path=config.checkpoint)

    # After training, evaluate the best checkpoint on the test split so runs are
    # directly comparable on held-out data. "best" points at the ModelCheckpoint's
    # top-k pick; if checkpointing is off it's empty, so fall back to final weights.
    if config.mode == "fit" and config.get("test_after_fit", False):
        best_ckpt = getattr(trainer.checkpoint_callback, "best_model_path", "") or None
        log.info(
            "Running test on %s checkpoint after fit.",
            "best" if best_ckpt else "final (no checkpoint saved)",
        )
        trainer.test(module, datamodule=datamodule, ckpt_path=best_ckpt)


if __name__ == "__main__":
    main()
