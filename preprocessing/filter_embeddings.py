"""Filter existing tile embeddings using saved tissue-selection and QC scores."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException


if TYPE_CHECKING:
    from collections.abc import Iterator


DEFAULT_DATA_MAPPING = "/mnt/projects/mammaprint/data_mapping.csv"
DEFAULT_EXPERIMENT = "MammaPrint"
DEFAULT_TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
SPLITS = ("train", "val", "test")
DEFAULT_STRATEGY_LOGICS = ("epithelium_only", "or", "and")
STRATEGY_LOGICS = (*DEFAULT_STRATEGY_LOGICS, "qc_only")
QC_MASKS = ("blur", "folding", "residual")
FILTER_STATE_FILENAME = "filter_state.json"
REQUIRED_SCORE_COLUMNS = ("slide_id", "x", "y")
REQUIRED_EMBEDDING_COLUMNS = ("x", "y", "embedding")


@dataclass(frozen=True)
class Strategy:
    slug: str
    logic: str


class FilterState(TypedDict):
    schema_version: int
    level: int
    tiling_uri: str
    embeddings_uri: str
    strategies: list[str]
    counts: dict[str, dict[str, int]]
    scored_tiles: int
    source_embedding_tiles: int
    epithelium_threshold: float
    cancer_threshold: float
    qc_thresholds: dict[str, float]


def _qc_thresholds(**thresholds: float | None) -> dict[str, float]:
    """Validate optional clean-area cutoffs; omission leaves that QC gate disabled."""
    unknown = set(thresholds) - set(QC_MASKS)
    if unknown:
        raise ValueError(f"Unknown QC masks: {sorted(unknown)}")
    active = {}
    for name in QC_MASKS:
        value = thresholds.get(name)
        if value is not None:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} threshold must be finite and in [0, 1]")
            active[name] = value
    return active


def _score_columns(
    selected_logics: tuple[str, ...], qc_thresholds: dict[str, float]
) -> tuple[str, ...]:
    if not selected_logics or set(selected_logics) - set(STRATEGY_LOGICS):
        raise ValueError(f"Invalid filtering strategies: {selected_logics}")
    if "qc_only" in selected_logics and not qc_thresholds:
        raise ValueError("qc_only requires at least one QC threshold")
    columns = []
    if set(selected_logics) & {"epithelium_only", "or", "and"}:
        columns.append("epithelium_overlap")
    if set(selected_logics) & {"or", "and"}:
        columns.append("cancer_overlap")
    columns.extend(f"{name}_overlap" for name in qc_thresholds)
    return tuple(columns)


def _validate_scores(scores: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(scores.columns))
    if missing:
        raise ValueError(f"Missing required mask-score columns: {missing}")
    for column in columns:
        values = pd.to_numeric(scores[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
            raise ValueError(f"{column} scores must be non-null, finite and in [0, 1]")


def _threshold_token(value: float) -> str:
    return f"{value:g}".replace(".", "")


def _strategies(
    epithelium_threshold: float,
    cancer_threshold: float,
    selected_logics: tuple[str, ...] = DEFAULT_STRATEGY_LOGICS,
    qc_thresholds: dict[str, float] | None = None,
) -> tuple[Strategy, ...]:
    qc = _qc_thresholds(**(qc_thresholds or {}))
    _score_columns(selected_logics, qc)
    epithelium = _threshold_token(epithelium_threshold)
    cancer = _threshold_token(cancer_threshold)
    definitions = (
        Strategy(f"epithel_{epithelium}", "epithelium_only"),
        Strategy(f"epithel_or_cancer_{epithelium}_{cancer}", "or"),
        Strategy(f"epithel_and_cancer_{epithelium}_{cancer}", "and"),
        Strategy("qc_only", "qc_only"),
    )
    selected = set(selected_logics)
    suffix = "".join(
        f"_{name}{str(float(value)).replace('.', '')}" for name, value in qc.items()
    )
    return tuple(
        Strategy(strategy.slug + suffix, strategy.logic)
        for strategy in definitions
        if strategy.logic in selected
    )


def _selection_masks(
    scores: pd.DataFrame,
    epithelium_threshold: float,
    cancer_threshold: float,
    selected_logics: tuple[str, ...] = DEFAULT_STRATEGY_LOGICS,
    qc_thresholds: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    qc = _qc_thresholds(**(qc_thresholds or {}))
    _validate_scores(scores, _score_columns(selected_logics, qc))
    clean = np.ones(len(scores), dtype=bool)
    for name, threshold in qc.items():
        clean &= scores[f"{name}_overlap"].to_numpy(dtype=float) > threshold
    result = {}
    for logic in selected_logics:
        selected = clean.copy()
        if logic != "qc_only":
            epithelium = (
                scores["epithelium_overlap"].to_numpy(dtype=float)
                > epithelium_threshold
            )
            if logic == "epithelium_only":
                selected &= epithelium
            else:
                cancer = (
                    scores["cancer_overlap"].to_numpy(dtype=float) > cancer_threshold
                )
                selected &= (
                    epithelium | cancer if logic == "or" else epithelium & cancer
                )
        result[logic] = selected
    return result


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], source: Path
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _load_split_map(data_mapping_path: Path) -> dict[str, str]:
    mapping = pd.read_csv(data_mapping_path)
    _require_columns(mapping, ("path", "split"), data_mapping_path)
    mapping = mapping[mapping["path"].notna() & mapping["split"].notna()]
    stems = mapping["path"].map(lambda path: Path(str(path)).stem)
    if stems.duplicated().any():
        duplicates = sorted(stems[stems.duplicated(keep=False)].unique())
        raise ValueError(
            f"Duplicate slide stems in {data_mapping_path}: {duplicates[:10]}"
        )

    split_map = dict(zip(stems, mapping["split"].astype(str), strict=True))
    unexpected = sorted(set(split_map.values()) - set(SPLITS))
    if unexpected:
        raise ValueError(
            f"Unexpected split values in {data_mapping_path}: {unexpected}"
        )
    return split_map


def _load_scores(
    tiling_dir: Path,
    expected_level: int,
    selected_logics: tuple[str, ...] = DEFAULT_STRATEGY_LOGICS,
    qc_thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, pd.DataFrame], int]:
    slides_path = tiling_dir / "slides.parquet"
    tiles_path = tiling_dir / "tiles.parquet"
    slides = pd.read_parquet(slides_path, columns=["id", "path", "level"])
    score_columns = _score_columns(
        selected_logics, _qc_thresholds(**(qc_thresholds or {}))
    )
    # Read only active mask columns: original cancer-only tiling has null epithelium.
    required_columns = (*REQUIRED_SCORE_COLUMNS, *score_columns)
    missing = sorted(set(required_columns) - set(pq.read_schema(tiles_path).names))
    if missing:
        raise ValueError(f"{tiles_path} is missing required columns: {missing}")
    tiles = pd.read_parquet(tiles_path, columns=list(required_columns))
    _require_columns(slides, ("id", "path", "level"), slides_path)
    _require_columns(tiles, REQUIRED_SCORE_COLUMNS, tiles_path)
    _validate_scores(tiles, score_columns)
    tiles = tiles[[*REQUIRED_SCORE_COLUMNS, *score_columns]]

    actual_levels = sorted(slides["level"].dropna().unique().tolist())
    if actual_levels != [expected_level]:
        raise ValueError(
            f"Expected a level-{expected_level} tiling artifact, found levels {actual_levels}"
        )

    if slides["id"].duplicated().any():
        raise ValueError(f"Duplicate slide IDs in {slides_path}")
    slides = slides.assign(stem=slides["path"].map(lambda path: Path(str(path)).stem))
    if slides["stem"].duplicated().any():
        duplicates = sorted(
            slides.loc[slides["stem"].duplicated(keep=False), "stem"].unique()
        )
        raise ValueError(f"Duplicate slide stems in {slides_path}: {duplicates[:10]}")

    id_to_stem = dict(zip(slides["id"], slides["stem"], strict=True))
    unknown_ids = sorted(set(tiles["slide_id"].unique()) - set(id_to_stem))
    if unknown_ids:
        raise ValueError(f"Tiles reference unknown slide IDs: {unknown_ids[:10]}")

    empty_scores = tiles.drop(columns="slide_id").iloc[:0].copy()
    by_stem = {stem: empty_scores.copy() for stem in slides["stem"]}
    for slide_id, frame in tiles.groupby("slide_id", sort=False):
        scores = frame.drop(columns="slide_id").reset_index(drop=True)
        if scores.duplicated(["x", "y"]).any():
            raise ValueError(
                f"Duplicate scored coordinates for slide {id_to_stem[slide_id]}"
            )
        by_stem[id_to_stem[slide_id]] = scores
    return by_stem, len(tiles)


def _prepare_output_dirs(
    root: Path, strategies: tuple[Strategy, ...]
) -> dict[str, dict[str, Path]]:
    output_dirs: dict[str, dict[str, Path]] = {}
    for strategy in strategies:
        directories = {"all": root / strategy.slug / "embeddings"}
        directories.update(
            {split: root / strategy.slug / f"embeddings_{split}" for split in SPLITS}
        )
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        output_dirs[strategy.logic] = directories
    return output_dirs


def _link_to_split(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        # A hard link can fail if a caller places temporary output across filesystems.
        shutil.copy2(source, destination)


def filter_embeddings(
    scores_by_stem: dict[str, pd.DataFrame],
    embeddings_dir: Path,
    output_dirs: dict[str, dict[str, Path]],
    split_map: dict[str, str],
    epithelium_threshold: float,
    cancer_threshold: float,
    logger: logging.Logger,
    qc_thresholds: dict[str, float] | None = None,
) -> tuple[dict[str, dict[str, int]], int]:
    qc = _qc_thresholds(**(qc_thresholds or {}))
    score_columns = _score_columns(tuple(output_dirs), qc)
    embedding_files = sorted(embeddings_dir.glob("*.parquet"))
    if not embedding_files:
        raise ValueError(f"No per-slide Parquet files found in {embeddings_dir}")

    embedding_stems = {path.stem for path in embedding_files}
    missing_embeddings = sorted(set(scores_by_stem) - embedding_stems)
    if missing_embeddings:
        raise ValueError(
            f"{len(missing_embeddings)} scored slides have no embedding file: "
            f"{missing_embeddings[:10]}"
        )
    unmapped_splits = sorted(embedding_stems - set(split_map))
    if unmapped_splits:
        raise ValueError(
            f"{len(unmapped_splits)} embedded slides have no split mapping: {unmapped_splits[:10]}"
        )

    counts = {
        logic: {"tiles": 0, "empty_slides": 0, "slides_below_100_tiles": 0}
        for logic in output_dirs
    }
    source_embedding_tiles = 0

    for index, embeddings_path in enumerate(embedding_files, start=1):
        stem = embeddings_path.stem
        source_embeddings = pd.read_parquet(embeddings_path)
        _require_columns(source_embeddings, REQUIRED_EMBEDDING_COLUMNS, embeddings_path)
        if source_embeddings.duplicated(["x", "y"]).any():
            raise ValueError(f"Duplicate embedded coordinates in {embeddings_path}")

        scores = scores_by_stem.get(stem)
        if scores is None:
            # Extra embedding slides are allowed only when the scored tiling set contains no tiles
            # for them. They are preserved as empty bags in every derived dataset.
            scores = pd.DataFrame(columns=["x", "y", *score_columns])

        merged = scores.merge(
            source_embeddings,
            on=["x", "y"],
            # QC-only must score every existing tile. A narrower or wrong tiling
            # must not silently discard source tiles and attribute that loss to QC.
            how="outer" if "qc_only" in output_dirs else "left",
            validate="one_to_one",
            indicator=True,
        )
        unscored = merged["_merge"].eq("right_only")
        if unscored.any():
            examples = merged.loc[unscored, ["x", "y"]].head().to_dict("records")
            raise ValueError(
                f"{int(unscored.sum())} embedded coordinates for {stem} have no QC scores: {examples}"
            )
        unmatched = merged["_merge"].eq("left_only")
        if unmatched.any():
            examples = merged.loc[unmatched, ["x", "y"]].head().to_dict("records")
            raise ValueError(
                f"{int(unmatched.sum())} scored coordinates for {stem} have no embedding: {examples}"
            )
        merged = merged.drop(columns="_merge")
        source_embedding_tiles += len(source_embeddings)
        masks = _selection_masks(
            merged, epithelium_threshold, cancer_threshold, tuple(output_dirs), qc
        )
        split = split_map[stem]

        for logic in output_dirs:
            selected = masks[logic]
            output = merged.loc[selected, list(REQUIRED_EMBEDDING_COLUMNS)]
            output_path = output_dirs[logic]["all"] / embeddings_path.name
            output.to_parquet(output_path, index=False, engine="pyarrow")
            _link_to_split(
                output_path, output_dirs[logic][split] / embeddings_path.name
            )
            counts[logic]["tiles"] += len(output)
            counts[logic]["empty_slides"] += int(output.empty)
            counts[logic]["slides_below_100_tiles"] += int(len(output) < 100)

        if index % 100 == 0 or index == len(embedding_files):
            logger.info("Filtered %d/%d slides", index, len(embedding_files))

    return counts, source_embedding_tiles


def _log_artifacts_with_retries(
    local_dir: Path,
    artifact_path: str,
    logger: logging.Logger,
    max_retries: int,
) -> None:
    for attempt in range(max_retries + 1):
        try:
            mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
            return
        except MlflowException:
            if attempt == max_retries:
                raise
            delay = 2**attempt
            logger.warning(
                "Uploading %s failed; retrying in %d second(s) (%d/%d)",
                artifact_path,
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)


def _log_outputs(
    *,
    level: int,
    tiling_uri: str,
    embeddings_uri: str,
    output_dirs: dict[str, dict[str, Path]],
    strategies: tuple[Strategy, ...],
    counts: dict[str, dict[str, int]],
    scored_tiles: int,
    source_embedding_tiles: int,
    epithelium_threshold: float,
    cancer_threshold: float,
    console_log: Path,
    logger: logging.Logger,
    upload_max_retries: int,
    qc_thresholds: dict[str, float] | None = None,
) -> None:
    qc = _qc_thresholds(**(qc_thresholds or {}))
    for strategy in strategies:
        dataset_name = f"l{level}_{strategy.slug}_embed"
        manifest = {
            "schema_version": 1,
            "dataset_name": dataset_name,
            "level": level,
            "strategy": strategy.logic,
            "epithelium_threshold": epithelium_threshold,
            "cancer_threshold": cancer_threshold,
            "qc_thresholds": qc,
            "comparison": ">",
            "source_tiling_uri": tiling_uri,
            "source_embeddings_uri": embeddings_uri,
            "scored_tiles": scored_tiles,
            "source_embedding_tiles": source_embedding_tiles,
            **counts[strategy.logic],
        }

        with mlflow.start_run(run_name=f"filter-{dataset_name}") as run:
            mlflow.log_params(
                {
                    "dataset_name": dataset_name,
                    "level": level,
                    "strategy": strategy.logic,
                    "epithelium_threshold": epithelium_threshold,
                    "cancer_threshold": cancer_threshold,
                    **{f"{name}_threshold": value for name, value in qc.items()},
                    "source_tiling_uri": tiling_uri,
                    "source_embeddings_uri": embeddings_uri,
                }
            )
            mlflow.set_tags(
                {
                    "data_variant": strategy.slug,
                    "source_tiling_uri": tiling_uri,
                    "source_embeddings_uri": embeddings_uri,
                }
            )
            mlflow.log_metrics(
                {key: float(value) for key, value in counts[strategy.logic].items()}
            )
            # Upload the training inputs first. If the optional combined artifact later
            # fails, the split datasets are already usable and can be resumed separately.
            for split_name in (*SPLITS, "all"):
                directory = output_dirs[strategy.logic][split_name]
                artifact_path = (
                    "embeddings" if split_name == "all" else f"embeddings_{split_name}"
                )
                logger.info("Logging %s to %s", dataset_name, artifact_path)
                _log_artifacts_with_retries(
                    directory,
                    artifact_path,
                    logger,
                    max_retries=upload_max_retries,
                )
            mlflow.log_dict(manifest, "filter_manifest.json")
            for handler in logger.handlers:
                handler.flush()
            experiment_id = run.info.experiment_id
            run_id = run.info.run_id
            logger.info("Created %s (run %s)", dataset_name, run_id)
            for split_name in ("all", *SPLITS):
                artifact_path = (
                    "embeddings" if split_name == "all" else f"embeddings_{split_name}"
                )
                logger.info(
                    "  %s: mlflow-artifacts:/%s/%s/artifacts/%s",
                    split_name,
                    experiment_id,
                    run_id,
                    artifact_path,
                )
            for handler in logger.handlers:
                handler.flush()
            mlflow.log_artifact(str(console_log))


@contextmanager
def _working_root(
    *, level: int, work_dir: Path, output_dir: Path | None
) -> Iterator[Path]:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        yield output_dir
        return

    with tempfile.TemporaryDirectory(
        prefix=f"filter-embeddings-l{level}-", dir=work_dir
    ) as tmp:
        yield Path(tmp)


def _state_payload(
    *,
    level: int,
    tiling_uri: str,
    embeddings_uri: str,
    strategies: tuple[Strategy, ...],
    counts: dict[str, dict[str, int]],
    scored_tiles: int,
    source_embedding_tiles: int,
    epithelium_threshold: float,
    cancer_threshold: float,
    qc_thresholds: dict[str, float] | None = None,
) -> FilterState:
    return {
        "schema_version": 1,
        "level": level,
        "tiling_uri": tiling_uri,
        "embeddings_uri": embeddings_uri,
        "strategies": [strategy.logic for strategy in strategies],
        "counts": counts,
        "scored_tiles": scored_tiles,
        "source_embedding_tiles": source_embedding_tiles,
        "epithelium_threshold": epithelium_threshold,
        "cancer_threshold": cancer_threshold,
        "qc_thresholds": _qc_thresholds(**(qc_thresholds or {})),
    }


def _load_filter_state(
    state_path: Path,
    *,
    level: int,
    tiling_uri: str,
    embeddings_uri: str,
    strategies: tuple[Strategy, ...],
    epithelium_threshold: float,
    cancer_threshold: float,
    qc_thresholds: dict[str, float] | None = None,
) -> FilterState:
    if not state_path.is_file():
        raise FileNotFoundError(
            f"Cannot use --upload-only because {state_path} does not exist"
        )
    state = cast("FilterState", json.loads(state_path.read_text()))
    expected = {
        "level": level,
        "tiling_uri": tiling_uri,
        "embeddings_uri": embeddings_uri,
        "epithelium_threshold": epithelium_threshold,
        "cancer_threshold": cancer_threshold,
    }
    mismatches = {
        key: (state.get(key), expected_value)
        for key, expected_value in expected.items()
        if state.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Saved filter state does not match this command: {mismatches}"
        )
    qc = _qc_thresholds(**(qc_thresholds or {}))
    # Legacy states predate QC filtering; they represent no additional QC gates.
    if state.get("qc_thresholds", {}) != qc:
        raise ValueError("Saved filter state QC thresholds do not match this command")
    requested_strategies = {strategy.logic for strategy in strategies}
    saved_strategies = set(state.get("strategies", []))
    missing_strategies = sorted(requested_strategies - saved_strategies)
    if missing_strategies:
        raise ValueError(
            f"Saved filter state does not contain requested strategies: {missing_strategies}"
        )
    return state


def _validate_output_dirs(output_dirs: dict[str, dict[str, Path]]) -> None:
    for logic, directories in output_dirs.items():
        all_files = {path.name: path for path in directories["all"].glob("*.parquet")}
        if not all_files:
            raise ValueError(f"No filtered Parquet files found for strategy {logic}")

        split_names: set[str] = set()
        for split in SPLITS:
            for split_path in directories[split].glob("*.parquet"):
                if split_path.name in split_names:
                    raise ValueError(
                        f"{split_path.name} appears in more than one split for {logic}"
                    )
                source_path = all_files.get(split_path.name)
                if source_path is None:
                    raise ValueError(
                        f"{split_path.name} appears in {split} but not in all for {logic}"
                    )
                if split_path.stat().st_size != source_path.stat().st_size:
                    raise ValueError(
                        f"Size mismatch for {logic}/{split}/{split_path.name}"
                    )
                split_names.add(split_path.name)

        missing_split_files = sorted(set(all_files) - split_names)
        if missing_split_files:
            raise ValueError(
                f"Filtered files have no split assignment for {logic}: "
                f"{missing_split_files[:10]}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter existing embeddings using saved epithelium/cancer and QC overlaps. "
            "Use --strategies qc_only to tighten QC on an already selected dataset."
        )
    )
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--tiling-uri", required=True)
    parser.add_argument("--embeddings-uri", required=True)
    parser.add_argument(
        "--embeddings-path",
        help="Existing mounted embedding directory; falls back to --embeddings-uri if absent.",
    )
    parser.add_argument("--epithelium-threshold", type=float, default=0.25)
    parser.add_argument("--cancer-threshold", type=float, default=0.5)
    for name in QC_MASKS:
        parser.add_argument(
            f"--{name}-threshold",
            type=float,
            help=f"Keep tiles with {name}_overlap > cutoff (clean fraction in [0, 1]); "
            "omitted means no additional gate for this mask.",
        )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGY_LOGICS,
        default=list(DEFAULT_STRATEGY_LOGICS),
        help="Strategies to materialise; defaults to the three epithelium/cancer strategies.",
    )
    parser.add_argument("--data-mapping", default=DEFAULT_DATA_MAPPING)
    parser.add_argument("--work-dir", default="/mnt/projects/mammaprint")
    parser.add_argument(
        "--output-dir",
        help=(
            "Persistent working directory. Filtered files and filter_state.json are kept "
            "here so a failed upload can be retried with --upload-only."
        ),
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Upload an existing --output-dir without filtering again.",
    )
    parser.add_argument("--upload-max-retries", type=int, default=3)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Keep results locally without creating MLflow runs; requires --output-dir.",
    )
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.upload_max_retries < 0:
        raise ValueError("--upload-max-retries must be non-negative")
    if args.upload_only and args.output_dir is None:
        raise ValueError("--upload-only requires --output-dir")
    if args.no_upload and (args.output_dir is None or args.upload_only):
        raise ValueError(
            "--no-upload requires --output-dir and cannot use --upload-only"
        )
    qc = _qc_thresholds(
        **{name: getattr(args, f"{name}_threshold") for name in QC_MASKS}
    )
    persistent_output = Path(args.output_dir) if args.output_dir else None
    if (
        persistent_output is not None
        and persistent_output.is_dir()
        and any(persistent_output.iterdir())
        and not args.upload_only
    ):
        raise FileExistsError(
            f"Persistent output directory is not empty: {persistent_output}. "
            "Use --upload-only to reuse completed filtering or choose a new directory."
        )

    selected_logics = tuple(dict.fromkeys(args.strategies))
    strategies = _strategies(
        args.epithelium_threshold,
        args.cancer_threshold,
        selected_logics,
        qc,
    )
    mlflow.set_tracking_uri(args.tracking_uri)
    if not args.no_upload:
        mlflow.set_experiment(args.experiment_name)

    with _working_root(
        level=args.level,
        work_dir=Path(args.work_dir),
        output_dir=persistent_output,
    ) as root:
        console_log = root / "console.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(console_log)],
            force=True,
        )
        logger = logging.getLogger(__name__)
        output_dirs = _prepare_output_dirs(root / "outputs", strategies)
        state_path = root / FILTER_STATE_FILENAME

        if args.upload_only:
            state = _load_filter_state(
                state_path,
                level=args.level,
                tiling_uri=args.tiling_uri,
                embeddings_uri=args.embeddings_uri,
                strategies=strategies,
                epithelium_threshold=args.epithelium_threshold,
                cancer_threshold=args.cancer_threshold,
                qc_thresholds=qc,
            )
            counts = state["counts"]
            scored_tiles = int(state["scored_tiles"])
            source_embedding_tiles = int(state["source_embedding_tiles"])
            logger.info("Reusing filtered output recorded in %s", state_path)
        else:
            logger.info("Downloading scored tiling artifact: %s", args.tiling_uri)
            tiling_download_dir = root / "source-tiling"
            tiling_download_dir.mkdir(exist_ok=True)
            tiling_dir = Path(
                download_artifacts(args.tiling_uri, dst_path=str(tiling_download_dir))
            )

            mounted_embeddings = (
                Path(args.embeddings_path) if args.embeddings_path is not None else None
            )
            if mounted_embeddings is not None and mounted_embeddings.is_dir():
                embeddings_dir = mounted_embeddings
                logger.info("Using mounted tissue embeddings: %s", embeddings_dir)
            else:
                if mounted_embeddings is not None:
                    logger.warning(
                        "Mounted embedding directory %s is unavailable; downloading %s",
                        mounted_embeddings,
                        args.embeddings_uri,
                    )
                else:
                    logger.info(
                        "Downloading tissue embedding artifact: %s", args.embeddings_uri
                    )
                embeddings_download_dir = root / "source-embeddings"
                embeddings_download_dir.mkdir(exist_ok=True)
                embeddings_dir = Path(
                    download_artifacts(
                        args.embeddings_uri,
                        dst_path=str(embeddings_download_dir),
                    )
                )

            scores_by_stem, scored_tiles = _load_scores(
                tiling_dir, args.level, selected_logics, qc
            )
            split_map = _load_split_map(Path(args.data_mapping))
            counts, source_embedding_tiles = filter_embeddings(
                scores_by_stem=scores_by_stem,
                embeddings_dir=embeddings_dir,
                output_dirs=output_dirs,
                split_map=split_map,
                epithelium_threshold=args.epithelium_threshold,
                cancer_threshold=args.cancer_threshold,
                logger=logger,
                qc_thresholds=qc,
            )
            state = _state_payload(
                level=args.level,
                tiling_uri=args.tiling_uri,
                embeddings_uri=args.embeddings_uri,
                strategies=strategies,
                counts=counts,
                scored_tiles=scored_tiles,
                source_embedding_tiles=source_embedding_tiles,
                epithelium_threshold=args.epithelium_threshold,
                cancer_threshold=args.cancer_threshold,
                qc_thresholds=qc,
            )
            state_path.write_text(json.dumps(state, indent=2))
            logger.info("Saved resumable filter state to %s", state_path)

        _validate_output_dirs(output_dirs)
        logger.info("Selection summary:\n%s", json.dumps(counts, indent=2))
        if args.no_upload:
            logger.info("Kept filtered embeddings locally in %s; upload disabled", root)
            return
        _log_outputs(
            level=args.level,
            tiling_uri=args.tiling_uri,
            embeddings_uri=args.embeddings_uri,
            output_dirs=output_dirs,
            strategies=strategies,
            counts=counts,
            scored_tiles=scored_tiles,
            source_embedding_tiles=source_embedding_tiles,
            epithelium_threshold=args.epithelium_threshold,
            cancer_threshold=args.cancer_threshold,
            console_log=console_log,
            logger=logger,
            upload_max_retries=args.upload_max_retries,
            qc_thresholds=qc,
        )


if __name__ == "__main__":
    main()
