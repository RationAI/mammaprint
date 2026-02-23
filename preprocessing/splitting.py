"""Split an existing tiling dataset into train/test_preliminary/test_final and upload to MLflow."""

from math import isclose

import hydra
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from sklearn.model_selection import train_test_split


def split_dataset(
    dataset: pd.DataFrame, splits: dict[str, float], random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a slide-level dataset into train / test_preliminary / test_final.

    The split is stratified by the ``type`` column.
    """
    assert isclose(
        splits["train"] + splits["test_preliminary"] + splits["test_final"], 1.0
    ), "Splits must sum to 1.0"

    if splits["train"] == 0.0:
        train = pd.DataFrame(columns=dataset.columns)
        test = dataset
    else:
        train, test = train_test_split(
            dataset,
            train_size=splits["train"],
            stratify=dataset["type"],
            random_state=random_state,
        )

    if splits["test_preliminary"] == 0.0:
        test_preliminary = pd.DataFrame(columns=test.columns)
        test_final = test
    else:
        preliminary_size = splits["test_preliminary"] / (1.0 - splits["train"])
        test_preliminary, test_final = train_test_split(
            test,
            train_size=preliminary_size,
            stratify=test["type"],
            random_state=random_state,
        )

    return train, test_preliminary, test_final


@with_cli_args(["+preprocessing=splitting"])
@hydra.main(
    config_path="../configs",
    config_name="ml",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # Download the source tiling dataset from MLflow
    dataset_dir = download_artifacts(config.source_dataset_uri)
    slides_df = pd.read_parquet(f"{dataset_dir}/slides.parquet")
    tiles_df = pd.read_parquet(f"{dataset_dir}/tiles.parquet")

    splits = {
        "train": config.splits.train,
        "test_preliminary": config.splits.test_preliminary,
        "test_final": config.splits.test_final,
    }

    # Split at slide level
    train_slides, test_preliminary_slides, test_final_slides = split_dataset(
        slides_df, splits, random_state=config.seed
    )

    print(
        f"\n[INFO] Split {len(slides_df)} slides: "
        f"train={len(train_slides)}, "
        f"test_preliminary={len(test_preliminary_slides)}, "
        f"test_final={len(test_final_slides)}\n"
    )

    # Assign tiles to splits based on slide_id
    for split_slides, split_name in [
        (train_slides, "train"),
        (test_preliminary_slides, "test_preliminary"),
        (test_final_slides, "test_final"),
    ]:
        split_tiles = tiles_df[tiles_df["slide_id"].isin(split_slides["id"])]

        print(
            f"[INFO] {split_name}: {len(split_slides)} slides, {len(split_tiles)} tiles"
        )

        save_mlflow_dataset(
            slides=split_slides,
            tiles=split_tiles,
            dataset_name=f"{config.data_name}_{split_name}",
        )

    print("\n[INFO] All splits saved to MLflow\n")


if __name__ == "__main__":
    main()
