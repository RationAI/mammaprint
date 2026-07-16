"""Entrypoint for MammaPrint MIL models — train / validate / test / predict.

Mirrors the preprocessing entrypoints: Hydra composes the config, ``@autolog``
sets up MLflow logging, and every object (datamodule, module, trainer) is built
with :func:`hydra.utils.instantiate`. The Lightning stage is selected by
``config.mode`` (``fit``/``validate``/``test``/``predict``), and ``config.checkpoint``
optionally resumes / loads weights.

Run with, e.g.::

    uv run -m ml.train +experiment=ml/train_mil_embeddings            # fit (default)
    uv run -m ml.train +experiment=ml/train_mil_embeddings mode=test checkpoint=/path/best.ckpt
"""

import random

import hydra
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


# Resolves ${random_seed:} in the config; cached so all consumers share one seed.
OmegaConf.register_new_resolver(
    "random_seed", lambda: random.randint(0, 2**31), use_cache=True, replace=True
)


@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # _recursive_=False: keep the per-split dataset nodes as DictConfig so the
    # DataModule can instantiate them lazily per stage in setup().
    datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)
    module = hydra.utils.instantiate(config.ml)
    trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # config.mode selects the Lightning stage (fit/validate/test/predict);
    # config.checkpoint (or null) resumes training / loads weights for eval.
    run_stage = getattr(trainer, config.mode)
    run_stage(module, datamodule=datamodule, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()
