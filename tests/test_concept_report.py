import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.concept_explainability.data import SlideInput
from ml.concept_explainability.report import (
    FORM_SCRIPT,
    render_naming_html,
    select_tiles,
    tile_geometry,
    write_presence_mask,
)


class ReportTests(unittest.TestCase):
    def test_inherited_mlflow_run_id_is_rejected_before_source_access(self):
        from unittest.mock import patch

        from omegaconf import OmegaConf

        from ml.concepts import run

        config = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "configs/concepts.yaml"
        )
        config.source_run_id = "source"
        with tempfile.TemporaryDirectory() as directory:
            config.output_dir = directory
            with (
                patch.dict("os.environ", {"MLFLOW_RUN_ID": "source"}),
                patch(
                    "ml.concepts.load_source_run",
                    side_effect=AssertionError("Source must not be accessed"),
                ),self.assertRaisesRegex(RuntimeError, "MLFLOW_RUN_ID")
            ):
                run(config)

    def test_actual_pyramid_level_and_level_relative_tile_coordinates(self):
        from unittest.mock import MagicMock, patch

        import openslide

        from ml.concept_explainability.report import read_rgb_tile, tile_viewport

        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.level_dimensions = [(4096, 2048), (1024, 512), (512, 256)]
        reader.level_downsamples = [1.0, 4.0, 8.0]
        reader.dimensions = (4096, 2048)
        reader.properties = {
            openslide.PROPERTY_NAME_MPP_X: "0.25",
            openslide.PROPERTY_NAME_MPP_Y: "0.25",
        }
        reader.read_region.return_value = Image.new("RGBA", (4, 4), (100, 50, 150, 128))
        slide = SlideInput(
            "s",
            np.zeros((1, 2560), np.float32),
            pd.DataFrame({"x": [8], "y": [4]}),
            np.zeros(1),
            Path("test.svs"),
            3,
            {"mpp": 1.0, "tile_extent": 4, "stride": 2},
        )
        with patch("openslide.OpenSlide", return_value=reader):
            geometry = tile_geometry(slide)
            self.assertEqual(geometry["scanner_level"], 1)
            self.assertEqual(geometry["downsample"], 4)
            image = read_rgb_tile(slide, 8, 4)
        reader.read_region.assert_called_once_with((32, 16), 1, (4, 4))
        self.assertEqual(image.getpixel((0, 0)), (177, 152, 202))
        viewport = tile_viewport(8, 4, geometry)
        self.assertAlmostEqual(viewport["point"]["x"], (8 + 2) / 1024)
        self.assertAlmostEqual(viewport["point"]["y"], (4 + 2) / 1024)

    def test_mask_coordinates_stay_in_selected_level(self):
        with tempfile.TemporaryDirectory() as directory:
            geometry = {
                "tile_extent": 4,
                "stride": 2,
                "width": 512,
                "height": 512,
                "downsample": 4,
                "mpp_x": 1.0,
                "mpp_y": 1.0,
            }
            path = Path(directory) / "mask.tiff"
            write_presence_mask(
                np.array([0.0, 1.0], np.float32),
                pd.DataFrame({"x": [0, 2], "y": [0, 0]}),
                geometry,
                0.0,
                1.0,
                path,
            )
            import pyvips

            image = pyvips.Image.new_from_file(str(path))
            pixels = np.ndarray(
                buffer=image.write_to_memory(),
                dtype=np.uint8,
                shape=(image.height, image.width),
            )
            self.assertEqual(pixels[0, 0], 1)
            self.assertIn(pixels[0, 2], [128, 129])
            self.assertEqual(pixels[0, 4], 255)

    def test_cli_fit_explain_end_to_end(self):
        from unittest.mock import patch

        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from test_concept_evaluation import fixture_source, local_download
        from test_concept_sources import BASE, tiny_config

        from ml.concepts import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture_source(root)
            source.config.ml = tiny_config().ml
            source.config.ml.encoder.out_dim = 2560
            source.config.ml.aggregator.feature_dim = 2560
            source.config.ml.head.in_dim = 2560
            for split in ("train", "val", "test"):
                for i in range(2):
                    Image.new("RGB", (512, 512), (100, 50, 150)).save(
                        root / f"{split}{i}.tiff"
                    )
            # The fixture slide has one level; do not relabel it as the real level-3 cohort.
            card = source.config.dataset.levels.pop("3")
            source.config.dataset.levels["0"] = card
            config_path = root / "source.yaml"
            OmegaConf.save(source.config, config_path)
            module = instantiate(source.config.ml).eval()
            checkpoint = root / "checkpoint.ckpt"
            torch.save({"state_dict": module.state_dict()}, checkpoint)
            before = checkpoint.read_bytes()
            config = OmegaConf.load(
                Path(__file__).resolve().parents[1] / "configs/concepts.yaml"
            )
            config.local_source_config = str(config_path)
            config.local_checkpoint = str(checkpoint)
            config.source_run_id = "offline-fixture"
            config.training_sha = BASE
            config.fixture_mode = True
            config.log_to_mlflow = False
            config.output_dir = str(root / "dictionary")
            config.fit.nmf.batch_size = 32
            config.fit.nmf.shuffle_buffer_size = 32
            with patch(
                "ml.data.datasets.single_scale.download_level_sources",
                side_effect=local_download,
            ):
                run(config)
                config.action = "explain"
                config.split = "val"
                config.dictionary_dir = str(root / "dictionary")
                config.output_dir = str(root / "report")
                run(config)
            report = root / "report" / "naming_k16.html"
            self.assertTrue(report.is_file())
            self.assertTrue(
                (root / "report" / "masks" / "concept_00" / "val0.tiff").exists()
            )
            self.assertIn("FIXTURE", report.read_text())
            self.assertEqual(before, checkpoint.read_bytes())
            self.assertEqual(
                pd.read_parquet(
                    root / "report" / "concept_effects.parquet"
                ).concept.nunique(),
                16,
            )

    def test_eight_examples_use_distinct_slides(self):
        keys = pd.DataFrame(
            {
                "slide_id": np.repeat(np.arange(10), 2),
                "row_index": np.tile([0, 1], 10),
                "x": np.zeros(20),
                "y": np.zeros(20),
            }
        )
        codes = np.ones((20, 16), np.float32)
        selected = select_tiles(codes, keys)
        self.assertEqual(len(selected), 16 * 8)
        self.assertTrue((selected.groupby("concept").slide_id.nunique() == 8).all())

    def test_insufficient_eligible_slides_is_reported(self):
        codes = np.vstack([np.zeros((8, 16)), np.ones((2, 16))])
        keys = pd.DataFrame(
            {
                "slide_id": list(map(str, range(10))),
                "row_index": np.zeros(10),
                "x": np.zeros(10),
                "y": np.zeros(10),
            }
        )
        selected = select_tiles(codes, keys)
        self.assertTrue((selected.groupby("concept").size() == 2).all())

    def test_blind_report_has_no_visible_labels_or_effects(self):
        selected = pd.DataFrame(
            {
                "concept": [0],
                "slide_id": ["<unsafe>"],
                "xopat_url": ["https://example.org"],
            }
        )
        html = render_naming_html(
            {"dictionary_id": "abc", "source_run_id": "run", "fixture_only": True},
            selected,
            ["aGVsbG8="],
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        self.assertIn('class="blind"', html)
        self.assertIn("body.blind .model-info", html)
        self.assertIn("fewer than 8", html)
        self.assertNotIn("<unsafe>", html)
        self.assertIn("FIXTURE", html)
        self.assertIn("Copy your answers", html)

    def test_reveal_preserves_answers_and_locks_expectation(self):
        # Execute the real report handlers with a tiny DOM double; no browser or network.
        harness = r"""
const state = {revealed: false};
const fields = Array.from({length:16}, (_, k) => ({value:'A', disabled:false, dataset:{concept:String(k)}}));
const other = {value:'name stays'};
const output = {value:'', hidden:true, select(){}};
global.document = {
  body:{classList:{remove(){state.revealed=true}}},
  querySelectorAll(q){return q==='.expected' ? fields : []},
  getElementById(id){if(id==='answers')return output; if(id==='reveal') return {disabled:false}; if(id==='copy-status') return {textContent:''}; if(id.startsWith('same-')) return {value:''}; return other}
};
global.navigator = {};
global.REPORT_ID = {dictionary_id:'dict', source_run_id:'run'};
"""
        assertions = r"""
revealResults();
if(!state.revealed || !fields.every(f=>f.disabled && f.value==='A')) throw Error('expectations changed');
copyAnswers().then(()=>{
 const data=JSON.parse(output.value);
 if(data.dictionary_id!=='dict'||data.answers.length!==16||!data.revealed)throw Error('invalid export');
 if(other.value!=='name stays')throw Error('other answers lost');
});
"""
        result = subprocess.run(
            ["node", "-e", harness + FORM_SCRIPT + assertions],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mask_and_tile_geometry_match_embedding_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "slide.tiff"
            Image.new("RGB", (512, 512), "white").save(path)
            keys = pd.DataFrame({"x": [0, 2], "y": [0, 0]})
            slide = SlideInput(
                "slide",
                np.zeros((2, 2560), np.float32),
                keys,
                np.zeros(1),
                path,
                0,
                {"mpp": 1, "tile_extent": 4, "stride": 2},
            )
            geometry = tile_geometry(slide)
            self.assertEqual(geometry["tile_extent"], 4)
            self.assertEqual(geometry["downsample"], 1)
            mask = root / "mask.tiff"
            write_presence_mask(
                np.array([0.0, 1.0], np.float32), keys, geometry, 0.0, 1.0, mask
            )
            import pyvips

            result = pyvips.Image.new_from_file(str(mask))
            pixels = np.ndarray(
                buffer=result.write_to_memory(),
                dtype=np.uint8,
                shape=(result.height, result.width),
            )
            self.assertEqual(pixels.shape, (512, 512))
            self.assertEqual(pixels[0, 0], 1)
            self.assertIn(pixels[0, 2], [128, 129])
            self.assertEqual(pixels[0, 4], 255)
            self.assertEqual(pixels[6, 10], 0)


if __name__ == "__main__":
    unittest.main()
