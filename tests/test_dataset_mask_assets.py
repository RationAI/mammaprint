from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import tifffile

from ml.explainability.dataset_mask_assets import prepare_mask_assets
from ml.explainability.dataset_mask_lineage import MaskSource


_RUN = "a" * 32
_WSI = "/mnt/slides/sample.tiff"


def _write_tiff(
    path: Path,
    pixels: np.ndarray,
    *,
    mpp: tuple[float, float] = (1.0, 1.2),
    pyramid: bool = False,
    orientation: int = 1,
    x_position: tuple[int, int] = (0, 1),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, ome=pyramid) as tiff:
        tiff.write(
            pixels,
            photometric="minisblack",
            resolution=(10000 / mpp[0], 10000 / mpp[1]),
            resolutionunit="CENTIMETER",
            tile=(256, 256) if pyramid else None,
            subifds=1 if pyramid else None,
            extratags=[
                (274, "H", 1, orientation, False),
                (286, "2I", 1, x_position, False),
            ],
        )
        if pyramid:
            tiff.write(
                pixels[::2, ::2],
                photometric="minisblack",
                resolution=(5000 / mpp[0], 5000 / mpp[1]),
                resolutionunit="CENTIMETER",
                tile=(256, 256),
                subfiletype=1,
            )


class _Client:
    def __init__(self, path: Path, experiment_id: str = "4") -> None:
        self.path = path
        self.experiment_id = experiment_id
        self.downloads: list[tuple[str, str]] = []

    def get_run(self, run_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, experiment_id=self.experiment_id)
        )

    def download_artifacts(self, run_id: str, artifact: str, dest: str) -> str:
        self.downloads.append((run_id, artifact))
        return str(shutil.copyfile(self.path, Path(dest) / Path(artifact).name))


def _source(location: str) -> MaskSource:
    return MaskSource(location=location, evidence="fixture", selection="test")


class DatasetMaskAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        _write_tiff(
            self.root / "slides/sample.tiff",
            np.zeros((600, 800), dtype=np.uint8),
            mpp=(0.5, 0.6),
        )

    def prepare(
        self,
        sources: dict[str, MaskSource],
        *,
        client: _Client | None = None,
        export: bool = False,
    ) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, dict[str, str]]]:
        return prepare_mask_assets(
            client,
            sources,
            {"slide-sample": _WSI},
            artifact_mount="secure_mlflow",
            wsi_root=self.root,
            mask_root=self.root,
            export_dir=self.root / "out" if export else None,
            export_viewer_prefix="private/registered-masks" if export else None,
        )

    def test_registered_originals_keep_immutable_source_references(self) -> None:
        pixels = np.zeros((300, 400), dtype=np.uint8)
        pixels[30:60, 100:125] = 255
        blur = self.root / "qc/Piqe_piqe_median_activity_mask_sample.tiff"
        cancer = self.root / "sample.tiff"
        _write_tiff(blur, pixels, pyramid=True)
        _write_tiff(cancer, pixels, pyramid=True)
        client = _Client(cancer)
        sources = {
            "blur": _source("/mnt/qc"),
            "cancer": _source(f"mlflow-artifacts:/4/{_RUN}/artifacts/heatmaps"),
            "folding": _source("/mnt/qc"),
        }

        assets, missing = self.prepare(sources, client=client)
        slide = assets["slide-sample"]
        self.assertEqual(
            slide["blur"]["viewer_path"],
            "qc/Piqe_piqe_median_activity_mask_sample.tiff",
        )
        self.assertEqual(
            slide["cancer"]["viewer_path"],
            f"secure_mlflow/4/{_RUN}/artifacts/heatmaps/sample.tiff",
        )
        self.assertEqual(
            slide["cancer"]["source"],
            f"mlflow-artifacts:/4/{_RUN}/artifacts/heatmaps/sample.tiff",
        )
        self.assertEqual(
            slide["cancer"]["sha256"], hashlib.sha256(cancer.read_bytes()).hexdigest()
        )
        self.assertEqual(client.downloads, [(_RUN, "heatmaps/sample.tiff")])
        self.assertIn("folding", missing["slide-sample"])

    def test_original_libtiff_pyramid_with_averaged_lower_level(self) -> None:
        path = self.root / "tissue/sample.tiff"
        path.parent.mkdir()
        pixels = np.zeros((600, 800), dtype=np.uint8)
        pixels[30:45, 50:61] = 255
        overview = np.zeros((300, 400), dtype=np.uint8)
        overview[15:23, 25:31] = 128
        with tifffile.TiffWriter(path, bigtiff=True) as tiff:
            for index, level in enumerate((pixels, overview)):
                tiff.write(
                    level,
                    photometric="minisblack",
                    resolution=(20000, 10000 / 0.6),
                    resolutionunit="CENTIMETER",
                    tile=(512, 512),
                    compression="deflate",
                    metadata=None,
                    subfiletype=int(index > 0),
                )
        assets, missing = self.prepare({"tissue": _source("/mnt/tissue")})
        self.assertFalse(missing)
        self.assertEqual(
            assets["slide-sample"]["tissue"]["viewer_path"], "tissue/sample.tiff"
        )

    def test_tiff_wsi_without_resolution_tags_uses_native_calibration(self) -> None:
        tiff_path = self.root / "slides/sample.tiff"
        tifffile.imwrite(tiff_path, np.zeros((600, 800), dtype=np.uint8))
        _write_tiff(
            self.root / "tissue/sample.tiff",
            np.zeros((300, 400), dtype=np.uint8),
            pyramid=True,
        )
        openslide = MagicMock()
        openslide.return_value.__enter__.return_value = SimpleNamespace(
            dimensions=(800, 600), slide_resolution=lambda level: (0.5, 0.6)
        )
        with patch.dict(
            sys.modules, {"ratiopath.openslide": SimpleNamespace(OpenSlide=openslide)}
        ):
            assets, missing = self.prepare({"tissue": _source("/mnt/tissue")})
        openslide.assert_called_once_with(str(tiff_path))
        self.assertIn("tissue", assets["slide-sample"])
        self.assertFalse(missing)

    def test_export_is_content_addressed_and_preserves_layer_semantics(self) -> None:
        pixels = np.zeros((300, 400), dtype=np.uint8)
        pixels[30, 101] = 1
        pixels[40:42, 110:112] = 17
        for kind in ("tissue", "cancer"):
            _write_tiff(self.root / f"{kind}/sample.tiff", pixels)
        sources = {kind: _source(f"/mnt/{kind}") for kind in ("tissue", "cancer")}

        assets, missing = self.prepare(sources, export=True)

        self.assertFalse(missing.get("slide-sample"))
        self.assertEqual(self.prepare(sources, export=True), (assets, missing))
        for kind, asset in assets["slide-sample"].items():
            file = self.root / "out" / Path(asset["viewer_path"]).name
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            self.assertEqual(
                asset["viewer_path"], f"private/registered-masks/{digest}.ome.tif"
            )
            self.assertEqual(asset["sha256"], digest)
            self.assertEqual(asset["source"], f"/mnt/{kind}/sample.tiff")
            with tifffile.TiffFile(file) as tiff:
                self.assertTrue(tiff.is_ome)
                levels = tiff.series[0].levels
                self.assertEqual(
                    [level.shape for level in levels], [(300, 400), (150, 200)]
                )
                expected = pixels if kind == "cancer" else (pixels > 0) * 255
                np.testing.assert_array_equal(levels[0].asarray(), expected)
                if kind == "tissue":
                    self.assertEqual(levels[1].asarray()[15, 50], 255)

    def test_refuses_unregistered_shapes_origins_and_orientations(self) -> None:
        for kind, mpp, orientation, position in (
            ("tissue", (1.1, 1.2), 1, (0, 1)),
            ("epithelium", (1.0, 1.2), 6, (0, 1)),
            ("cancer", (1.0, 1.2), 1, (1, 10)),
        ):
            _write_tiff(
                self.root / f"{kind}/sample.tiff",
                np.zeros((300, 400), dtype=np.uint8),
                mpp=mpp,
                orientation=orientation,
                x_position=position,
            )
        sources = {
            kind: _source(f"/mnt/{kind}") for kind in ("tissue", "epithelium", "cancer")
        }
        assets, missing = self.prepare(sources, export=True)
        self.assertFalse(assets.get("slide-sample"))
        self.assertEqual(set(missing["slide-sample"]), set(sources))
        for kind, reason in {
            "tissue": "extent",
            "epithelium": "orientation",
            "cancer": "origin",
        }.items():
            self.assertIn(reason, missing["slide-sample"][kind].lower())

    def test_does_not_download_from_mismatched_artifact_experiment(self) -> None:
        client = _Client(self.root / "never-read.tiff", experiment_id="8")
        assets, missing = self.prepare(
            {"cancer": _source(f"mlflow-artifacts:/4/{_RUN}/artifacts/heatmaps")},
            client=client,
        )
        self.assertFalse(assets.get("slide-sample"))
        self.assertIn("experiment", missing["slide-sample"]["cancer"])
        self.assertFalse(client.downloads)

    def test_no_export_mount_reports_unsupported_original(self) -> None:
        _write_tiff(
            self.root / "tissue/sample.tiff", np.zeros((300, 400), dtype=np.uint8)
        )
        assets, missing = self.prepare({"tissue": _source("/mnt/tissue")})
        self.assertFalse(assets.get("slide-sample"))
        self.assertIn("export", missing["slide-sample"]["tissue"].lower())
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
