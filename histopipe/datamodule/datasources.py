# Standard Imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

# Third-party Imports
import mlflow
import pandas
from sklearn.model_selection import train_test_split


class BaseDataSource(ABC):
    """Abstract class for DataSource.

    It defines required methods and contains some fields that are common to all data
    sources.
    """

    splits: dict[str, float] | None
    stratified_keys: list[str] | None
    seed: int

    def __init__(
        self,
        seed: int,
        splits: dict[str, float] | None,
        stratified_keys: list[str] | None,
    ) -> None:
        self.seed = seed
        self.splits = self.__validate_splits(splits)
        self.stratified_keys = self.__validate_stratified_keys(stratified_keys)

    @staticmethod
    def __validate_splits(splits: dict[str, float] | None) -> dict[str, float] | None:
        if splits is None:
            return None

        if not (isinstance(splits, dict) or len(splits) > 0):
            raise ValueError(
                "splits must be a dictionary of split names and fractions or None"
            )

        return splits

    @staticmethod
    def __validate_stratified_keys(
        stratified_keys: list[str] | None,
    ) -> list[str] | None:
        if stratified_keys is None:
            return None
        elif isinstance(stratified_keys, list) and len(stratified_keys) > 0:
            return stratified_keys
        else:
            raise ValueError(
                "stratified_keys must be a list of column identifiers or None"
            )

    @abstractmethod
    def get_table(self) -> pandas.DataFrame:
        """Retrieves full dataset defined by this data source.

        Returns:
            pandas.DataFrame: Full dataset.
        """

    @abstractmethod
    def get_metadata(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Retrieves metadata for a dataset of entries defined by `data`.

        It is expected the `data` is a subset of the full dataset defined by this data
        source. The metadata values are retrieved based on a foreign key relationship
        between `data` and the metadata table.

        Args:
            data (pandas.DataFrame): Data to retrieve metadata for.

        Returns:
            pandas.DataFrame: New dataframe with metadata columns.
        """

    @abstractmethod
    def split(self) -> dict[str, Self] | Self:
        """Splits datasource into N parts, where N is `len(splits)`.

        The size of each split is defined by the values in the `splits` dictionary.
        The keys are used to name the splits.

        Returns:
            dict[str, BaseDataSource] | BaseDataSource: split data sources.
        """


class WSIDataSource(BaseDataSource):
    __slide_foreign_key = "slide_name"  # Foreign key is hardcoded
    tiles: pandas.DataFrame
    slides: pandas.DataFrame

    def __init__(
        self,
        seed: int,
        splits: dict[str, float] | None = None,
        stratified_keys: list[str] | None = None,
        data_uris: str | list[str] | None = None,
    ):
        super().__init__(seed, splits, stratified_keys)
        if data_uris is not None:
            data_uris = self.__validate_data_uris(data_uris)
            self.slides, self.tiles = self.load_mlflow_artifacts(data_uris)

    # Access Methods

    def get_table(self) -> pandas.DataFrame:
        return self.tiles

    def get_metadata(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Retrieves metadata for a dataframe using natural join on a foreign key."""
        return self.slides.join(
            data.set_index(self.__slide_foreign_key),
            on=self.__slide_foreign_key,
            how="inner",
        )

    # Mlflow Methods

    @staticmethod
    def load_mlflow_artifacts(
        data_uris: list[str],
    ) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        slides_dfs = []
        tiles_dfs = []
        for uri in data_uris:
            try:
                fp = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            except mlflow.MlflowException as e:
                raise FileNotFoundError(f"Cannot fetch data from {uri}.") from e

            try:
                slides = pandas.read_parquet(fp / "slides.parquet")
                tiles = pandas.read_parquet(fp / "tiles.parquet")
            except OSError as e:
                raise FileNotFoundError(f"Cannot load data from {fp}.") from e

            slides_dfs.append(slides)
            tiles_dfs.append(tiles)

        slides_df, tiles_df = pandas.concat(slides_dfs), pandas.concat(tiles_dfs)

        # Select only specific slides
        # slides_df = slides_df[slides_df['slide_name'].isin(['fll-003','skn-002','kdn-002'])]

        return slides_df, tiles_df

    @staticmethod
    def __validate_data_uris(data_uris):
        if data_uris is None:
            return []
        elif not isinstance(data_uris, list):
            raise TypeError(f"Data URIs must be a list, got {type(data_uris)}")

        for uri in data_uris:
            if not isinstance(uri, str):
                raise TypeError(f"Data URI must be a string, got {type(uri)}")
            if not uri.startswith("mlflow-artifacts:/"):
                raise ValueError(
                    f"Data URI must start with 'mlflow-artifacts:/', got {uri}"
                )
        return data_uris

    # Splitting methods
    def split(self) -> dict[Self]:
        """Split dataset into train and validation."""
        # Could also be test - name is NOT declared
        if self.splits is None:
            raise ValueError(
                f"Splits in config set as {self.splits}."
                "Impossible for Pytorch Lightning to determine which stage "
                "(train|valid|test) should this data be used."
            )

        if len(self.splits) == 1:
            k, _ = dict(self.splits).popitem()
            return {k: self}

        # If multiple keys are specified for stratification then concat the values
        if self.stratified_keys is not None:
            stratify = (
                self.slides[self.stratified_keys].astype(str).agg("-".join, axis=1)
            )
        else:
            stratify = None

        # Split dataframes
        (
            train_slides,
            train_tiles,
            valid_slides,
            valid_tiles,
        ) = WSIDataSource._split_data(
            slides=self.slides,
            tiles=self.tiles,
            train_size=self.splits["train"],
            test_size=None,
            stratify=stratify,
            seed=self.seed,
        )

        # Create and return new WSIDataSource objects
        train_datasource = WSIDataSource._create_split_datasource(
            train_slides, train_tiles, self.seed
        )
        valid_datasource = WSIDataSource._create_split_datasource(
            valid_slides, valid_tiles, self.seed
        )
        return {"train": train_datasource, "valid": valid_datasource}

    @classmethod
    def _create_split_datasource(cls, slides, tiles, seed) -> Self:
        """Creates a new WSIDataSource instance with supplied slides and tiles."""
        datasource = WSIDataSource(
            seed=seed, splits=None, stratified_keys=None, data_uris=None
        )
        datasource.slides = slides
        datasource.tiles = tiles
        return datasource

    @staticmethod
    def _split_data(
        slides: pandas.DataFrame,
        tiles: pandas.DataFrame,
        train_size: float | int | None = None,
        test_size: float | int | None = None,
        stratify: list | pandas.Series | None = None,
        seed: int | None = None,
    ) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        """Splits slides/tiles into train/valid splits of requested sizes.

        Allows stratification based on columns (or combinations of columns) such that
        the occurence of each unique value of that column or their combinations is at
        least 2.
        """
        train_slides, valid_slides = train_test_split(
            slides,
            train_size=train_size,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            stratify=stratify,
        )
        train_tiles = WSIDataSource._filter_tiles_by_slides(tiles, train_slides)
        valid_tiles = WSIDataSource._filter_tiles_by_slides(tiles, valid_slides)
        return train_slides, train_tiles, valid_slides, valid_tiles

    @staticmethod
    def _filter_tiles_by_slides(
        tiles: pandas.DataFrame, slides: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Filters out all tiles from those WSIs that do not appear in the provided slides dataframe."""
        all_slide_names = tiles[WSIDataSource.__slide_foreign_key]
        requested_slide_names = slides[WSIDataSource.__slide_foreign_key]
        filter_mask = all_slide_names.isin(requested_slide_names)
        filtered_tiles = tiles[filter_mask]
        return filtered_tiles


class JpgDataSource(BaseDataSource):
    """Data source for jpg images.

    File structure of jpg data source should follows:
        labels.parquet: dataframe with labels
        <name>.jpg:     input images
    """

    table: pandas.DataFrame

    def __init__(
        self,
        seed: int,
        splits: dict[str, float] | None = None,
        data_uri: str | None = None,
    ) -> None:
        super().__init__(seed, splits, None)
        if data_uri is not None:
            self.path = self.download_mlflow_artifacts(data_uri)
            self.table = self.load_table(self.path)

    @staticmethod
    def download_mlflow_artifacts(data_uri: str) -> Path:
        try:
            return Path(mlflow.artifacts.download_artifacts(data_uri))

        except mlflow.MlflowException as e:
            raise FileNotFoundError(f"cannot fetch data from {data_uri}") from e

    @staticmethod
    def load_table(path: Path) -> pandas.DataFrame:
        return pandas.read_parquet(path / "labels.parquet")

    def get_table(self) -> pandas.DataFrame:
        return self.table

    def get_metadata(self, data: pandas.DataFrame) -> pandas.DataFrame:
        return self.table

    def get_path(self) -> Path:
        return self.path

    def split(self) -> dict[str, Self]:
        if self.splits is None:
            raise ValueError(
                f"Splits in config set as {self.splits}."
                "Impossible for Pytorch Lightning to determine which stage "
                "(train|valid|test) should this data be used."
            )

        if len(self.splits) == 1:
            k, _ = dict(self.splits).popitem()
            return {k: self}

        train, valid = train_test_split(
            self.table,
            train_size=self.splits["train"],
            random_state=self.seed,
            shuffle=True,
        )

        return {
            "train": JpgDataSource.create_datasource(train, self.path, self.seed),
            "valid": JpgDataSource.create_datasource(valid, self.path, self.seed),
        }

    @classmethod
    def create_datasource(cls, table: pandas.DataFrame, path: Path, seed: int) -> Self:
        ds = JpgDataSource(seed)
        ds.table = table
        ds.path = path
        return ds
