"""Analyze cancer-mask tile scores and choose a production filtering threshold.

This is the non-interactive cluster version of mask_coverage_threshold.ipynb.
It is preconfigured for the corrected level-3 cancer-probability tiling run.
The script prints quantiles and exact keep/drop counts, then logs CSV tables and
a histogram/CDF figure to a small MLflow analysis run.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mlflow.artifacts import download_artifacts


DEFAULT_TRACKING_URI = "http://mlflow-s3.rationai-mlflow"
DEFAULT_DATASET_URI = (
    "mlflow-artifacts:/3/41a0cfd625574ed9a5c547a8c254bc8a/artifacts/"
    "mou_3_224_cancer_probability_threshold_scan"
)
DEFAULT_COLUMN = "cancer_overlap"
DEFAULT_THRESHOLDS = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
QUANTILES = (
    0.0,
    0.001,
    0.01,
    0.02,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999,
    1.0,
)
LUMINAL_MAP = {"a luminal": "A", "b luminal": "B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cancer_overlap from a tiled MLflow dataset.",
    )
    parser.add_argument("--dataset-uri", default=DEFAULT_DATASET_URI)
    parser.add_argument("--column", default=DEFAULT_COLUMN)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--experiment", default="MammaPrint")
    parser.add_argument(
        "--run-name",
        default="Cancer probability threshold analysis l3",
    )
    return parser.parse_args()


def resolve_tiles_path(dataset_uri: str) -> Path:
    local_path = Path(download_artifacts(dataset_uri))
    if local_path.is_file():
        return local_path
    tiles_path = local_path / "tiles.parquet"
    if not tiles_path.is_file():
        raise FileNotFoundError(
            f"Downloaded '{dataset_uri}', but '{tiles_path}' does not exist."
        )
    return tiles_path


def load_tiles(tiles_path: Path, column: str) -> pd.DataFrame:
    available_columns = set(pq.read_schema(tiles_path).names)
    if column not in available_columns:
        raise ValueError(
            f"Column '{column}' is missing from {tiles_path}. "
            f"Available columns: {sorted(available_columns)}"
        )

    columns = [column]
    if "type" in available_columns:
        columns.append("type")
    tiles = pd.read_parquet(tiles_path, columns=columns, engine="pyarrow")
    tiles[column] = pd.to_numeric(tiles[column], errors="coerce")
    if "type" in tiles:
        tiles["luminal_class"] = (
            tiles["type"].astype("string").str.strip().str.lower().map(LUMINAL_MAP)
        )
    return tiles


def quantile_table(tiles: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.Series]] = [("all", tiles[column])]
    if "luminal_class" in tiles:
        groups.extend(
            (f"luminal_{label}", group[column])
            for label, group in tiles.groupby("luminal_class", dropna=True)
        )

    for group_name, values in groups:
        clean = values.dropna()
        quantiles = clean.quantile(QUANTILES)
        for quantile, value in quantiles.items():
            rows.append(
                {
                    "group": group_name,
                    "quantile": float(quantile),
                    "cancer_probability": float(value),
                    "tile_count": len(clean),
                }
            )
    return pd.DataFrame(rows)


def threshold_table(
    tiles: pd.DataFrame,
    column: str,
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.Series]] = [("all", tiles[column])]
    if "luminal_class" in tiles:
        groups.extend(
            (f"luminal_{label}", group[column])
            for label, group in tiles.groupby("luminal_class", dropna=True)
        )

    for group_name, values in groups:
        clean = values.dropna()
        total = len(clean)
        for threshold in thresholds:
            kept = int((clean > threshold).sum())
            dropped = total - kept
            rows.append(
                {
                    "group": group_name,
                    "threshold": threshold,
                    "total_tiles": total,
                    "kept_tiles": kept,
                    "dropped_tiles": dropped,
                    "kept_percent": 100.0 * kept / total if total else 0.0,
                    "dropped_percent": 100.0 * dropped / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def class_bias_table(thresholds: pd.DataFrame) -> pd.DataFrame:
    class_rows = thresholds[thresholds["group"].isin(["luminal_A", "luminal_B"])]
    if class_rows["group"].nunique() != 2:
        return pd.DataFrame()
    pivot = class_rows.pivot(
        index="threshold",
        columns="group",
        values="kept_percent",
    ).reset_index()
    pivot["absolute_keep_rate_gap_pp"] = (
        pivot["luminal_A"] - pivot["luminal_B"]
    ).abs()
    return pivot


def plot_distribution(
    tiles: pd.DataFrame,
    column: str,
    thresholds: list[float],
    output_path: Path,
) -> None:
    bins = np.linspace(0.0, 1.0, 201)
    figure, (histogram_axis, cdf_axis) = plt.subplots(1, 2, figsize=(14, 5))

    groups: list[tuple[str, pd.Series]] = [("all", tiles[column])]
    if "luminal_class" in tiles:
        class_groups = [
            (f"luminal {label}", group[column])
            for label, group in tiles.groupby("luminal_class", dropna=True)
        ]
        if class_groups:
            groups = class_groups

    for label, values in groups:
        clean = values.dropna().to_numpy()
        counts, edges = np.histogram(clean, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        density = counts / counts.sum() / np.diff(edges) if counts.sum() else counts
        cdf = np.cumsum(counts) / counts.sum() if counts.sum() else counts
        histogram_axis.plot(centers, density, label=label)
        cdf_axis.plot(edges[1:], cdf, label=label)

    for threshold in thresholds:
        histogram_axis.axvline(threshold, color="black", alpha=0.18, linewidth=0.7)
        cdf_axis.axvline(threshold, color="black", alpha=0.18, linewidth=0.7)

    histogram_axis.set(
        title=f"{column}: probability distribution",
        xlabel="mean cancer probability",
        ylabel="density",
        xlim=(0.0, 1.0),
    )
    cdf_axis.set(
        title="CDF: fraction dropped at threshold",
        xlabel="threshold",
        ylabel="fraction with score ≤ threshold",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    histogram_axis.legend()
    cdf_axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    mlflow.set_tracking_uri(args.tracking_uri)

    print(f"Downloading tiled dataset: {args.dataset_uri}")
    tiles_path = resolve_tiles_path(args.dataset_uri)
    tiles = load_tiles(tiles_path, args.column)
    values = tiles[args.column].dropna()

    print(f"Loaded {len(tiles):,} tile rows from {tiles_path}")
    print(f"Valid {args.column} values: {len(values):,}")
    print(f"Unique {args.column} values: {values.nunique():,}")
    print("\nCancer probability summary:")
    print(
        values.describe(
            percentiles=[0.001, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
        ).to_string()
    )

    quantiles = quantile_table(tiles, args.column)
    thresholds = threshold_table(tiles, args.column, args.thresholds)
    bias = class_bias_table(thresholds)

    print("\nThreshold keep/drop table (all tiles):")
    print(
        thresholds[thresholds["group"] == "all"].to_string(
            index=False,
            formatters={
                "kept_percent": "{:.2f}".format,
                "dropped_percent": "{:.2f}".format,
            },
        )
    )
    if not bias.empty:
        print("\nLuminal A/B keep-rate comparison:")
        print(bias.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    mlflow.set_experiment(args.experiment)
    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory)
        quantiles.to_csv(output_dir / "cancer_probability_quantiles.csv", index=False)
        thresholds.to_csv(output_dir / "cancer_threshold_keep_drop.csv", index=False)
        if not bias.empty:
            bias.to_csv(output_dir / "cancer_threshold_class_bias.csv", index=False)
        plot_distribution(
            tiles,
            args.column,
            args.thresholds,
            output_dir / "cancer_probability_distribution_cdf.png",
        )

        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.log_param("source_dataset_uri", args.dataset_uri)
            mlflow.log_param("score_column", args.column)
            mlflow.log_param("candidate_thresholds", ",".join(map(str, args.thresholds)))
            mlflow.log_metric("tile_count", len(values))
            mlflow.log_metric("cancer_probability_mean", float(values.mean()))
            mlflow.log_metric("cancer_probability_median", float(values.median()))
            mlflow.log_artifacts(str(output_dir), artifact_path="threshold_analysis")
            print(f"\nLogged analysis to MLflow run: {run.info.run_id}")
            print(
                "Analysis artifacts: "
                f"mlflow-artifacts:/{run.info.experiment_id}/{run.info.run_id}/"
                "artifacts/threshold_analysis"
            )


if __name__ == "__main__":
    main()
