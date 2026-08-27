import unittest
from pathlib import Path
from typing import ClassVar

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


CONFIGS = Path(__file__).parents[2] / "configs"


class MultiscaleExperimentTest(unittest.TestCase):
    EXPERIMENTS: ClassVar[dict[str, dict[str, object]]] = {
        "multiscale_mean_linear": {
            "run_name": "Multiscale mean/concat/mean + linear — L5/L4/L3",
            "stages": (None, "mean", "concat", "mean", None),
            "slide_dim": 7680,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 7681,
        },
        "multiscale_level_mean_linear": {
            "run_name": "Multiscale level mean + linear — L5/L4/L3",
            "stages": (None, "mean", "mean", "mean", None),
            "slide_dim": 2560,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 2561,
        },
        "multiscale_raw_flat_linear": {
            "run_name": "Multiscale raw 21-slot flatten + linear — L5/L4/L3",
            "stages": (None, None, None, "mean", "flatten"),
            "slide_dim": 53760,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 53761,
        },
        "multiscale_mean_mlp": {
            "run_name": "Multiscale mean/concat/mean + MLP — L5/L4/L3",
            "stages": (None, "mean", "concat", "mean", None),
            "slide_dim": 7680,
            "head": "ml.models.heads.MLPHead",
            "trainable_parameters": 1966593,
        },
        "multiscale_projected_mean": {
            "run_name": "Multiscale projected mean + linear — L5/L4/L3",
            "stages": ("shared_linear", "mean", "concat", "mean", None),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 1328129,
        },
        "multiscale_slide_attention": {
            "run_name": "Multiscale projected mean + slide attention — L5/L4/L3",
            "stages": (
                "shared_linear",
                "mean",
                "concat",
                "gated_attention",
                None,
            ),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 2118402,
        },
        "multiscale_local_attention": {
            "run_name": "Multiscale local + slide attention — L5/L4/L3",
            "stages": (
                "shared_linear",
                "gated_attention",
                "concat",
                "gated_attention",
                None,
            ),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 2382339,
        },
        "multiscale_transformer_mean_slide": {
            "run_name": "Multiscale local Transformer + mean slide — L5/L4/L3",
            "stages": (
                "shared_linear",
                "shared_transformer",
                "concat",
                "mean",
                None,
            ),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 4493313,
        },
        "multiscale_transformer": {
            "run_name": "Multiscale local Transformer + slide attention — L5/L4/L3",
            "stages": (
                "shared_linear",
                "shared_transformer",
                "concat",
                "gated_attention",
                None,
            ),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 5283586,
        },
        "multiscale_joint_transformer": {
            "run_name": "Multiscale joint 21-token Transformer — L5/L4/L3",
            "stages": (
                "shared_linear",
                None,
                "joint_transformer",
                "gated_attention",
                None,
            ),
            "slide_dim": 512,
            "head": "ml.models.heads.LinearHead",
            "trainable_parameters": 4756226,
        },
    }

    def test_all_experiments_compose_common_training_and_resolved_metadata(
        self,
    ) -> None:
        for experiment, expected in self.EXPERIMENTS.items():
            with self.subTest(experiment=experiment):
                with initialize_config_dir(
                    version_base=None,
                    config_dir=str(CONFIGS),
                ):
                    config = compose(
                        config_name="ml",
                        overrides=[f"+experiment=ml/multiscale/{experiment}"],
                    )

                self.assertEqual(
                    [
                        config.datamodule.train.split,
                        config.datamodule.val.split,
                        config.datamodule.test.split,
                    ],
                    ["train", "val", "test"],
                )
                self.assertEqual(
                    {
                        config.datamodule.train._target_,
                        config.datamodule.val._target_,
                        config.datamodule.test._target_,
                    },
                    {"ml.data.PyramidScaleDataset"},
                )
                self.assertEqual(set(map(int, config.dataset.levels)), {3, 4, 5})
                self.assertEqual(config.ml.loss._target_, "torch.nn.BCEWithLogitsLoss")
                self.assertEqual(config.ml.optimizer._target_, "torch.optim.Adam")
                self.assertTrue(config.ml.optimizer._partial_)
                self.assertEqual(
                    config.ml.lr_scheduler.scheduler._target_,
                    "torch.optim.lr_scheduler.ReduceLROnPlateau",
                )
                self.assertEqual(
                    set(config.ml.metrics),
                    {"accuracy", "auroc", "f1", "precision", "recall", "specificity"},
                )
                self.assertIs(config.test_after_fit, False)
                self.assertEqual(config.seed, 1729)

                metadata = OmegaConf.to_container(
                    config.metadata.hyperparams,
                    resolve=True,
                )
                self.assertEqual(config.metadata.run_name, expected["run_name"])
                self.assertEqual(
                    metadata,
                    {
                        "tile_extent": 224,
                        "overlap_policy": "non_overlapping",
                        "selection_seed": 0,
                        "feature_adapter": expected["stages"][0],
                        "within_level": expected["stages"][1],
                        "level_fusion": expected["stages"][2],
                        "slide_pool": expected["stages"][3],
                        "finalizer": expected["stages"][4],
                        "feature_dim": 2560,
                        "hidden_dim": 512,
                        "slide_dim": expected["slide_dim"],
                        "head": expected["head"],
                        "loss": "torch.nn.BCEWithLogitsLoss",
                        "trainable_parameters": expected["trainable_parameters"],
                        "training_seed": 1729,
                    },
                )

                module = instantiate(config.ml)
                self.assertEqual(
                    sum(
                        parameter.numel()
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ),
                    expected["trainable_parameters"],
                )


if __name__ == "__main__":
    unittest.main()
