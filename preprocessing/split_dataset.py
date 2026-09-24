"""Materialise per-split MLflow artifacts from a whole dataset artifact."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.artifacts import download_artifacts
from rationai.tiling.writers import save_mlflow_dataset


DATA_MAPPING = "/mnt/projects/mammaprint/data_mapping.csv"
MLFLOW_EXPERIMENT = "MammaPrint"
DEFAULT_TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
SPLITS = ("train", "val", "test")


def _load_split_map(data_mapping_path: str) -> dict[str, str]:
    """Return ``slide_stem -> split`` from data_mapping.csv, keyed by path stem."""
    df = pd.read_csv(data_mapping_path)
    if "split" not in df.columns:
        raise ValueError(f"'{data_mapping_path}' has no 'split' column.")

    df = df[df["path"].notna() & df["split"].notna()]
    stems = df["path"].apply(lambda path: Path(str(path)).stem)
    split_map = dict(zip(stems, df["split"].astype(str), strict=True))
    bad = {split for split in split_map.values() if split not in SPLITS}
    if bad:
        raise ValueError(f"Unexpected split values in '{data_mapping_path}': {bad}")
    return split_map


def _assign_or_die(stems: list[str], split_map: dict[str, str]) -> dict[str, str]:
    """Map each artifact stem to its split; abort if any stem is unmapped."""
    unmapped = sorted(stem for stem in stems if stem not in split_map)
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} slide(s) in the artifact have no split row in "
            f"data_mapping.csv: {unmapped}. Add them (or fix the artifact) first."
        )
    return {stem: split_map[stem] for stem in stems}


def _configure_mlflow() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"Using MLflow tracking URI: {tracking_uri} (experiment: {MLFLOW_EXPERIMENT})")


def _uri_from_card(card_path: str) -> str:
    """Read the whole artifact URI from a tiled or embedded level card."""
    card = yaml.safe_load(Path(card_path).read_text())
    uris = card.get("raw_uris") or card.get("uris") or {}
    all_uri = uris.get("all")
    if not all_uri:
        raise ValueError(
            f"Card '{card_path}' has no `raw_uris.all` / `uris.all` URI to split."
        )
    return all_uri


def _name_from_uri(uri: str) -> str:
    name = Path(uri.rstrip("/")).name
    if name in {"", ".", "artifacts"}:
        raise ValueError(f"Could not infer a dataset name from --uri '{uri}'. Pass --name.")
    return name


def _split_embedded(local_dir: Path, name: str, split_map: dict[str, str]) -> None:
    files = sorted(local_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No <stem>.parquet files found under '{local_dir}'.")
    assignment = _assign_or_die([file.stem for file in files], split_map)

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        split_dirs = {split: Path(tmp) / split for split in SPLITS}
        for directory in split_dirs.values():
            directory.mkdir()
        for file in files:
            shutil.copy2(file, split_dirs[assignment[file.stem]] / file.name)

        for split in SPLITS:
            count = sum(1 for value in assignment.values() if value == split)
            dataset_name = f"{name}_{split}"
            with mlflow.start_run(run_name=f"split-{dataset_name}"):
                mlflow.log_artifacts(str(split_dirs[split]), dataset_name)
            print(f"  {split}: {count} slides -> {dataset_name}")


def _split_tiled(local_dir: Path, name: str, split_map: dict[str, str]) -> None:
    slides = pd.read_parquet(local_dir / "slides.parquet")
    tiles = pd.read_parquet(local_dir / "tiles.parquet")
    slides = slides.assign(_stem=slides["path"].apply(lambda path: Path(str(path)).stem))
    assignment = _assign_or_die(slides["_stem"].tolist(), split_map)
    slides = slides.assign(split=slides["_stem"].map(assignment))

    for split in SPLITS:
        split_slide_ids = slides.loc[slides["split"] == split, "id"]
        split_slides = slides[slides["split"] == split].drop(columns=["_stem", "split"])
        split_tiles = tiles[tiles["slide_id"].isin(split_slide_ids)]
        dataset_name = f"{name}_{split}"
        with mlflow.start_run(run_name=f"split-{dataset_name}"):
            save_mlflow_dataset(
                slides=split_slides.reset_index(drop=True),
                tiles=split_tiles.reset_index(drop=True),
                dataset_name=dataset_name,
            )
        print(
            f"  {split}: {len(split_slides)} slides, {len(split_tiles)} tiles "
            f"-> {dataset_name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialise per-split MLflow artifacts from data_mapping.csv."
    )
    parser.add_argument("--kind", choices=["embedded", "tiled"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--card", help="Level card whose `all` URI is the artifact to split.")
    source.add_argument("--uri", help="Whole artifact URI (overrides --card).")
    parser.add_argument("--name", help="Output base name (default: inferred from URI).")
    parser.add_argument("--data-mapping", default=DATA_MAPPING)
    args = parser.parse_args()

    split_map = _load_split_map(args.data_mapping)
    _configure_mlflow()
    uri = args.uri or _uri_from_card(args.card)
    name = args.name or _name_from_uri(uri)
    local_dir = Path(download_artifacts(artifact_uri=uri))
    print(f"Splitting {args.kind} artifact '{uri}' as '{name}_{{train,val,test}}'")

    if args.kind == "embedded":
        _split_embedded(local_dir, name, split_map)
    else:
        _split_tiled(local_dir, name, split_map)

    print(
        "Done. Point the level card at the three MLflow artifacts named:\n"
        f"  {name}_train\n  {name}_val\n  {name}_test"
    )


if __name__ == "__main__":
    main()
