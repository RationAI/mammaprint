"""Verify final embedding cards compose with every modular architecture."""

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ml.data.datasets.pyramid import PyramidLayout


CONFIGS = Path(__file__).parents[2] / "configs"
FAMILIES = (
    "tissue_only",
    "cancer_mask_5",
    "cancer_mask_9",
    "epithel",
    "epithel_and_cancer_5",
    "epithel_or_cancer_5",
)
WINDOWS = ((5, 4, 3), (4, 3, 2))


class FinalPyramidCardsTest(unittest.TestCase):
    def test_all_final_families_and_windows_reuse_original_cards(self) -> None:
        experiments = sorted(
            path.stem
            for path in (CONFIGS / "experiment/ml/multiscale").glob("*.yaml")
            if not path.stem.startswith("_")
        )
        with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
            originals = {
                (family, level): OmegaConf.to_container(
                    compose(
                        config_name=f"data/embedded/{family}/l{level}"
                    ).dataset.levels[str(level)],
                    resolve=True,
                )
                for family in FAMILIES
                for level in (2, 3, 4, 5)
            }
            for family in FAMILIES:
                for levels in WINDOWS:
                    card = family + "_" + "_".join(f"l{level}" for level in levels)
                    for experiment in experiments:
                        with self.subTest(card=card, experiment=experiment):
                            config = compose(
                                config_name="ml",
                                overrides=[
                                    f"+experiment=ml/multiscale/{experiment}",
                                    f"data/pyramid={card}",
                                ],
                            )
                            self.assertEqual(config.pyramid.dataset_family, family)
                            self.assertEqual(
                                config.metadata.hyperparams.dataset_family, family
                            )
                            self.assertEqual(
                                tuple(config.metadata.hyperparams.level_order), levels
                            )
                            self.assertTrue(
                                config.metadata.run_name.endswith(
                                    "/".join(f"L{level}" for level in levels)
                                )
                            )
                            layout = PyramidLayout.from_levels(
                                config.dataset.levels,
                                anchor_level=config.pyramid.anchor_level,
                                level_order=config.pyramid.level_order,
                                tile_extent=config.pyramid.tile_extent,
                            )
                            self.assertEqual(layout.level_order, levels)
                            self.assertEqual(layout.ratios, (1, 2, 4))
                            self.assertEqual(layout.token_count, 21)
                            self.assertEqual(
                                tuple(config.ml.aggregator.level_ratios),
                                layout.ratios,
                            )
                            self.assertEqual(
                                set(map(int, config.dataset.levels)), set(levels)
                            )
                            for level in levels:
                                self.assertEqual(
                                    OmegaConf.to_container(
                                        config.dataset.levels[str(level)], resolve=True
                                    ),
                                    originals[family, level],
                                )
                            for split in ("train", "val", "test"):
                                dataset = config.datamodule[split]
                                self.assertEqual(dataset.split, split)
                                self.assertEqual(tuple(dataset.level_order), levels)
                                self.assertEqual(
                                    tuple(dataset.level_ratios), layout.ratios
                                )
                                for level in levels:
                                    self.assertEqual(
                                        dataset.levels[str(level)].uris[split],
                                        config.dataset.levels[str(level)].uris[split],
                                    )


if __name__ == "__main__":
    unittest.main()
