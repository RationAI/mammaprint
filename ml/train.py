"""Training entrypoint for MammaPrint MIL models.

Mirrors the preprocessing entrypoints: Hydra composes the config, ``@autolog``
sets up MLflow logging, and every object (datamodule, module, trainer) is built
with :func:`hydra.utils.instantiate`.

Run with, e.g.::

    uv run -m ml.train +experiment=ml/train_mil_embeddings
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
    datamodule = hydra.utils.instantiate(config.datamodule)
    module = hydra.utils.instantiate(config.ml)
    trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
