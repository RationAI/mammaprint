import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIGS = Path(__file__).parents[2] / "configs"


class MultiscaleConfigTest(unittest.TestCase):
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
