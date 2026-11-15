import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate


CONFIGS = Path(__file__).parents[2] / "configs"
PRESET_OVERRIDES = [
    "+feature_dim=2560",
    "+multiscale_hidden_dim=512",
    "+pyramid.level_ratios=[1,2,4]",
]


class MultiscaleConfigTest(unittest.TestCase):
    PRESETS: ClassVar[dict[str, dict[str, object]]] = {
        "multiscale_mean_linear": {
            "stages": (None, "mean", "concat", "mean", None),
            "slide_dim": 7680,
            "head": "ml.models.heads.LinearHead",
        },
        "multiscale_level_mean_linear": {
            "stages": (None, "mean", "mean", "mean", None),
            "slide_dim": 2560,
            "head": "ml.models.heads.LinearHead",
        },
        "multiscale_raw_flat_linear": {
            "stages": (None, None, None, "mean", "flatten"),
            "slide_dim": 53760,
            "head": "ml.models.heads.LinearHead",
        },
        "multiscale_mean_mlp": {
            "stages": (None, "mean", "concat", "mean", None),
            "slide_dim": 7680,
            "head": "ml.models.heads.MLPHead",
        },
        "multiscale_projected_mean": {
            "stages": ("shared_linear", "mean", "concat", "mean", None),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
        },
        "multiscale_slide_attention": {
            "stages": (
                "shared_linear",
                "mean",
                "concat",
                "gated_attention",
                None,
            ),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
        },
        "multiscale_local_attention": {
            "stages": (
                "shared_linear",
                "gated_attention",
                "concat",
                "gated_attention",
                None,
            ),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
        },
        "multiscale_transformer_mean_slide": {
            "stages": (
                "shared_linear",
                "shared_transformer",
                "concat",
                "mean",
                None,
            ),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
        },
        "multiscale_transformer": {
            "stages": (
                "shared_linear",
                "shared_transformer",
                "concat",
                "gated_attention",
                None,
            ),
            "slide_dim": 1536,
            "head": "ml.models.heads.LinearHead",
        },
        "multiscale_joint_transformer": {
            "stages": (
                "shared_linear",
                None,
                "joint_transformer",
                "gated_attention",
                None,
            ),
            "slide_dim": 512,
            "head": "ml.models.heads.LinearHead",
        },
    }

    def test_named_presets_compose_and_match_their_head_widths(self) -> None:
        for preset, expected in self.PRESETS.items():
            with self.subTest(preset=preset):
                with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
                    config = compose(
                        config_name=f"ml/aggregator/multiscale_presets/{preset}",
                        overrides=PRESET_OVERRIDES,
                    )

                aggregator_config = config.ml.aggregator
                self.assertEqual(
                    tuple(
                        aggregator_config[stage]
                        for stage in (
                            "feature_adapter",
                            "within_level",
                            "level_fusion",
                            "slide_pool",
                            "finalizer",
                        )
                    ),
                    expected["stages"],
                )
                self.assertEqual(config.slide_dim, expected["slide_dim"])
                self.assertEqual(config.ml.head._target_, expected["head"])

                aggregator = instantiate(aggregator_config)
                head = instantiate(config.ml.head)
                self.assertEqual(aggregator.out_dim, head.in_dim)
                self.assertEqual(aggregator.out_dim, config.slide_dim)

    def test_hydra_null_overrides_are_python_none_and_invalid_programs_explain_why(
        self,
    ) -> None:
        cases = (
            ("feature_adapter", "null", None),
            ("within_level", "null", None),
            ("level_fusion", "null", None),
            ("slide_pool", "null", None),
            ("finalizer", "null", None),
            ("position_embedding", "null", None),
            ("level_embedding", "null", None),
        )
        for key, override, expected in cases:
            with self.subTest(key=key):
                with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
                    config = compose(
                        config_name="ml/aggregator/multiscale_presets/multiscale_mean_linear",
                        overrides=[
                            *PRESET_OVERRIDES,
                            f"ml.aggregator.{key}={override}",
                        ],
                    )
                self.assertIs(config.ml.aggregator[key], expected)

        with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
            invalid = compose(
                config_name="ml/aggregator/multiscale_presets/multiscale_mean_linear",
                overrides=[
                    *PRESET_OVERRIDES,
                    "ml.aggregator.within_level=null",
                ],
            )
        with self.assertRaisesRegex(
            InstantiationException,
            "Rejected stage 'level_fusion'.*pyramid_tokens.*accepted alternatives",
        ):
            instantiate(invalid.ml.aggregator)

        with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
            invalid_embeddings = compose(
                config_name="ml/aggregator/multiscale_presets/multiscale_mean_linear",
                overrides=[
                    *PRESET_OVERRIDES,
                    "ml.aggregator.position_embedding=learned",
                ],
            )
        with self.assertRaisesRegex(
            InstantiationException,
            "Position and level embeddings are only available for Transformer stages",
        ):
            instantiate(invalid_embeddings.ml.aggregator)

    def test_dataset_rejects_layout_overrides_that_disagree_with_model(self) -> None:
        cases = (
            "pyramid.level_order=[5,3,4]",
            "pyramid.level_ratios=[1,4,2]",
            "ml.aggregator.level_ratios=[1,4,2]",
            "dataset.levels.4.mpp=1.94",
        )
        for override in cases:
            for split in ("train", "val", "test"):
                with self.subTest(override=override, split=split):
                    with initialize_config_dir(
                        version_base=None, config_dir=str(CONFIGS)
                    ):
                        config = compose(
                            config_name="ml",
                            overrides=[
                                "+experiment=ml/multiscale/multiscale_mean_linear",
                                override,
                            ],
                        )
                    with (
                        patch(
                            "ml.data.datasets.pyramid.download_level_sources"
                        ) as download,
                        self.assertRaisesRegex(
                            InstantiationException, "but the model expects level_ratios"
                        ),
                    ):
                        instantiate(config.datamodule[split])
                    download.assert_not_called()

    def test_multiscale_experiment_composes_existing_level_cards(self) -> None:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIGS),
        ):
            config = compose(
                config_name="ml",
                overrides=["+experiment=ml/train_mil_embeddings_multiscale"],
            )

        self.assertEqual(
            config.datamodule.train._target_, "ml.data.PyramidScaleDataset"
        )
        self.assertEqual(list(config.pyramid.level_order), [5, 4, 3])
        self.assertEqual(list(config.pyramid.level_ratios), [1, 2, 4])
        self.assertEqual(config.pyramid.tile_extent, 224)
        self.assertEqual(
            config.ml.aggregator._target_,
            "ml.models.aggregators.MultiScaleMIL",
        )
        self.assertEqual(set(map(int, config.dataset.levels)), {3, 4, 5})
        self.assertEqual(config.dataset.levels["3"].name, "l3_tissue_embed")
        self.assertEqual(config.dataset.levels["4"].name, "l4_tissue_embed")
        self.assertEqual(config.dataset.levels["5"].name, "l5_tissue_embed")


if __name__ == "__main__":
    unittest.main()
