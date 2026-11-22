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

import hydra
import numpy as np
import torch
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

from ml.data import DataModule, SingleScaleDataset
from ml.models import MammaprintModule
from ml.models.heads import LinearHead


log = logging.getLogger(__name__)


# Resolves ${random_seed:} in the config; cached so all consumers share one seed.
OmegaConf.register_new_resolver(
    "random_seed", lambda: random.randint(0, 2**31), use_cache=True, replace=True
)


@torch.no_grad()
def _initialize_index_head(
    datamodule: DataModule, module: MammaprintModule, logger: MLFlowLogger
) -> None:
    datamodule.setup("fit")
    dataset = datamodule.train_dataset
    head = module.head
    if not isinstance(dataset, SingleScaleDataset) or not isinstance(head, LinearHead):
        raise ValueError(
            "Index head initialization requires a single-scale dataset and linear head."
        )
    if head.out_dim != 1:
        raise ValueError("Index head initialization requires a single output.")

    targets = dataset.regression_targets()
    target_mean = targets.mean()
    head.linear.bias.fill_(target_mean.item())

    # Use a fixed set of slides and an independent tile sample for the probe.
    indices = np.linspace(0, len(dataset) - 1, num=min(16, len(dataset)), dtype=int)
    rng = np.random.default_rng(0)
    was_training = module.training
    module.eval()
    try:
        predictions = torch.stack(
            [module(dataset.load_sample(int(idx), rng=rng)[0])[0] for idx in indices]
        ).flatten()
    finally:
        module.train(was_training)
    logger.log_metrics(
        {
            "init/target_mean": target_mean.item(),
            "init/target_sd": targets.std(unbiased=False).item(),
            "init/prediction_mean": predictions.mean().item(),
            "init/prediction_sd": predictions.std(unbiased=False).item(),
        },
        step=0,
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

    if (
        config.mode == "fit"
        and config.checkpoint is None
        and config.get("initialize_index_head", False)
    ):
        _initialize_index_head(datamodule, module, logger)

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
