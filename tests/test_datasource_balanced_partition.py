# Standard Imports
from pathlib import Path

import hydra
import pandas

# Third Party Imports
from omegaconf import omegaconf

# Local Imports
from histopipe.datamodule.datasources import WSIDataSource


class CSVDataSource(WSIDataSource):
    """DataSource for loading CSV Storage Files.

    There are two CSV files: one for the tiles and one for the slides.
    The `slides.csv` file contains the metadata for each slide.
    The `tiles.csv` file contains the tile coordinates and the slide_id.
    """

    __slide_fk_key = "slide_id"

    def __init__(
        self,
        seed: int,
        data_fp: str,
        splits: dict[str, float] | None = None,
        stratified_keys: list[str] | None = None,
    ) -> None:
        super().__init__(seed=seed, splits=splits, stratified_keys=stratified_keys)
        self.load_data_from_dir(Path(data_fp))

    def load_data_from_dir(self, fp: Path) -> None:
        self.slides = pandas.read_parquet(fp / "slides.csv")
        self.tiles = pandas.read_parquet(fp / "tiles.csv")


def test_class_fraction_invariant_across_partitions_simple():
    ds_cfg = omegaconf.OmegaConf.load("conf/datasource_simple.yml")
    ds = hydra.utils.instantiate(ds_cfg, _convert_="partial")
    split_data_sources = ds.split()

    # check that there is a correct number of splits
    assert len(split_data_sources) == len(ds.splits)

    # check that no records are lost
    all_records = len(ds.tiles)
    for _split_name, ds_split in split_data_sources.items():
        all_records -= len(ds_split.tiles)
    assert all_records == 0, "Records are not distributed correctly between splits."
