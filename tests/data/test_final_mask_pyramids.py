"""Exercise final mask-filter outputs through pyramid datasets and prediction."""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from preprocessing.filter_embeddings import (
    _prepare_output_dirs,
    _strategies,
    filter_embeddings,
)


CONFIGS = Path(__file__).parents[2] / "configs"
FAMILY_LOGIC = {
    "epithel": "epithelium_only",
    "epithel_and_cancer_5": "and",
    "epithel_or_cancer_5": "or",
}


class FinalMaskPyramidsTest(unittest.TestCase):
    def test_filtered_split_outputs_feed_mean_and_transformer_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.csv"
            split_map = {
                f"{kind}_{split}": split
                for split in ("train", "val", "test")
                for kind in ("complete", "incomplete", "empty")
            }
            pd.DataFrame(
                {
                    "path": [f"/slides/{name}.svs" for name in split_map],
                    "type": ["a luminal"] * len(split_map),
                    "mammaprint_index": [0.5] * len(split_map),
                    "split": list(split_map.values()),
                }
            ).to_csv(mapping, index=False)
            outputs = self._filter_levels(root, split_map)

            for family, logic in FAMILY_LOGIC.items():
                for levels in ((5, 4, 3), (4, 3, 2)):
                    for preset in ("multiscale_mean_linear", "multiscale_transformer"):
                        card = family + "_" + "_".join(f"l{level}" for level in levels)
                        with self.subTest(card=card, preset=preset):
                            with initialize_config_dir(
                                version_base=None, config_dir=str(CONFIGS)
                            ):
                                config = compose(
                                    config_name="ml",
                                    overrides=[
                                        f"+experiment=ml/multiscale/{preset}",
                                        f"data/pyramid={card}",
                                        f"dataset.data_mapping={mapping}",
                                        "feature_dim=8",
                                        "multiscale_hidden_dim=8",
                                        "slide_dim=24",
                                    ],
                                )
                            model = instantiate(config.ml).eval()
                            for split in ("train", "val", "test"):
                                with patch(
                                    "ml.data.datasets.pyramid.download_level_sources",
                                    return_value={
                                        level: outputs[level][logic][split]
                                        for level in levels
                                    },
                                ) as download:
                                    dataset = instantiate(config.datamodule[split])
                                download.assert_called_once_with(
                                    {
                                        level: config.dataset.levels[str(level)].uris[
                                            split
                                        ]
                                        for level in levels
                                    }
                                )
                                audit = dataset.audit_frame().set_index("slide_id")
                                self.assertEqual(
                                    set(audit.index),
                                    {
                                        f"{kind}_{split}"
                                        for kind in ("complete", "incomplete", "empty")
                                    },
                                )
                                self.assertTrue(audit.loc[f"empty_{split}", "rejected"])
                                self.assertEqual(
                                    audit.loc[f"incomplete_{split}", "retained"],
                                    0 if levels[0] == 5 else 3,
                                )
                                self.assertEqual(
                                    len(dataset), 1 if levels[0] == 5 else 2
                                )
                                samples = [
                                    dataset[index] for index in range(len(dataset))
                                ]
                                for bag, _, metadata in samples:
                                    self.assertEqual(tuple(bag.shape[1:]), (21, 8))
                                    self.assertTrue(
                                        metadata["slide_id"].endswith(split)
                                    )
                                    self.assertEqual(
                                        len(metadata["anchors"]), bag.shape[0]
                                    )
                                with torch.no_grad():
                                    prediction = model.predict_step(
                                        samples, batch_idx=0
                                    )
                                self.assertEqual(
                                    tuple(prediction["predictions"].shape),
                                    (len(dataset), 1),
                                )
                                self.assertTrue(
                                    torch.isfinite(prediction["predictions"]).all()
                                )
                                self.assertEqual(
                                    prediction["metadata"],
                                    [sample[2] for sample in samples],
                                )

    @staticmethod
    def _filter_levels(root: Path, split_map: dict[str, str]):
        outputs = {}
        for level in (5, 4, 3, 2):
            source = root / f"source_l{level}"
            source.mkdir()
            width = 2 ** (5 - level)
            points = [
                (column * 224, row * 224)
                for row in range(width)
                for column in range(width)
            ]
            scores = {}
            for slide_id in split_map:
                frame = pd.DataFrame(
                    {
                        "x": [point[0] for point in points],
                        "y": [point[1] for point in points],
                        "embedding": [
                            np.arange(8, dtype=np.float32) + level for _ in points
                        ],
                    }
                )
                frame.to_parquet(source / f"{slide_id}.parquet", index=False)
                keep = np.ones(len(points), dtype=float)
                if slide_id.startswith("empty"):
                    keep[:] = 0.0
                elif slide_id.startswith("incomplete") and level in (3, 2):
                    keep[0] = 0.0
                scores[slide_id] = frame[["x", "y"]].assign(
                    epithelium_overlap=keep, cancer_overlap=keep
                )
            outputs[level] = _prepare_output_dirs(
                root / f"filtered_l{level}", _strategies(0.25, 0.5)
            )
            filter_embeddings(
                scores_by_stem=scores,
                embeddings_dir=source,
                output_dirs=outputs[level],
                split_map=split_map,
                epithelium_threshold=0.25,
                cancer_threshold=0.5,
                logger=logging.getLogger(__name__),
            )
        return outputs


if __name__ == "__main__":
    unittest.main()
