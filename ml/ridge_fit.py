"""Fit a closed-form CPU ridge model to one embedding source."""

from __future__ import annotations

import csv
import fcntl
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np
from sklearn.model_selection import GroupKFold

from ml.ridge_experiment import signed_read, signed_write
from ml.ridge_screening import (
    Cohort,
    Feature,
    FeatureStore,
    Protocol,
    fit_ridge,
    ridge_path,
)
from preprocessing.slide_statistics import (
    SOURCE_KEY,
    build_cache,
    digest,
    file_digest,
    load_manifest,
    load_membership,
    source_inventory,
)


if TYPE_CHECKING:
    from preprocessing.slide_statistics import Slide


FEATURE_COLUMNS = {
    "raw-mean": ("mean_raw",),
    "ln-mean": ("mean_ln",),
    "raw-mean-std": ("mean_raw", "std_raw"),
    "ln-mean-std": ("mean_ln", "std_ln"),
}


@dataclass(frozen=True)
class FitOptions:
    mapping: Path
    level: int
    mask: str
    feature: str
    target_column: str
    embedding_dim: int
    alpha: float | None
    folds: int
    seed: int
    source: str | None
    cards_dir: Path
    cache_dir: Path | None
    scratch_dir: Path | None

    @property
    def key(self) -> str:
        key = f"l{self.level}/{self.mask}"
        if not SOURCE_KEY.fullmatch(key):
            raise ValueError(
                "Expected a positive level and a lowercase mask name such as tissue_only"
            )
        return key

    def validate(self) -> None:
        _ = self.key
        if not self.target_column.strip() or self.target_column in {
            "path",
            "record_num",
            "split",
        }:
            raise ValueError("Expected a separate continuous target column")
        if self.embedding_dim < 1:
            raise ValueError("--embedding-dim must be positive")
        if self.alpha is not None and (
            not math.isfinite(self.alpha) or self.alpha <= 0
        ):
            raise ValueError("--alpha must be positive and finite")
        if self.folds < 2 or not 0 <= self.seed < 2**32:
            raise ValueError("Expected at least two folds and a seed in [0, 2^32)")


def resolve_source(options: FitOptions) -> dict[str, Any]:
    if options.source is None:
        inventory = source_inventory({"cards_dir": options.cards_dir})
    else:
        uri = options.source
        if not urlsplit(uri).scheme:
            uri = str(Path(uri).resolve())
        inventory = source_inventory({"sources": {options.key: uri}})
    entry = inventory.get(options.key)
    if entry is None or entry["status"] != "available":
        raise ValueError(
            f"No embedding source for {options.key}. Supply --source or select an available card."
        )
    if "card" in entry and not urlsplit(entry["uri"]).scheme:
        card_directory = (options.cards_dir / entry["card"]).parent
        entry = {**entry, "uri": str((card_directory / entry["uri"]).resolve())}
    return entry


def development_mapping(path: Path, target_column: str) -> dict[str, Slide]:
    slides = load_membership(path)
    development = {key: slide for key, slide in slides.items() if slide.split != "test"}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if target_column not in (rows.fieldnames or ()):
            raise ValueError(f"Mapping lacks target column {target_column!r}")
        for row in rows:
            if row["split"] == "test" or not row["path"] or not row["path"].strip():
                continue
            key = Path(row["path"]).stem
            try:
                value = float(row[target_column])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                development[key] = replace(development[key], target=value)
    return development


def write_development_membership(path: Path, slides: dict[str, Slide]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("path", "record_num", "split"))
        for key, slide in sorted(slides.items()):
            writer.writerow((f"{key}.slide", slide.record_num, slide.split))


def fit_standalone(
    options: FitOptions, output: Path, source: dict[str, Any] | None
) -> Path:
    """Save a final ridge fit with provenance and no test scoring."""
    mapping_sha = file_digest(options.mapping)
    slides = development_mapping(options.mapping, options.target_column)
    labeled = {key for key, slide in slides.items() if slide.target is not None}
    if not labeled:
        raise ValueError(f"No finite development targets in {options.target_column!r}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".write.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache_dir = options.cache_dir if source is None else output / "cache"
        if cache_dir is None:
            raise ValueError("A cache directory or an embedding source is required")
        with tempfile.TemporaryDirectory(prefix="ridge-fit-membership-") as workdir:
            membership = Path(workdir) / "development.csv"
            write_development_membership(membership, slides)
            membership_sha = file_digest(membership)
            if source is not None:
                source_options = {"uri": source["uri"]}
                provenance = {
                    name: source[name]
                    for name in ("card", "card_sha256")
                    if name in source
                }
                if provenance:
                    source_options["provenance"] = provenance
                build_cache(
                    {
                        "data_mapping": membership,
                        "sources": {options.key: source_options},
                        "embedding_dim": options.embedding_dim,
                        "scratch_dir": options.scratch_dir,
                    },
                    cache_dir,
                )
        manifest = load_manifest(cache_dir)
        if manifest.get("pilot") or manifest.get("pilot_slides") is not None:
            raise ValueError("A full cache is required for fitting")
        if manifest["mapping_sha256"] not in (mapping_sha, membership_sha):
            raise ValueError("Data mapping does not match the cache")
        return train_model(
            options, output, cache_dir, manifest, mapping_sha, slides, labeled
        )


def train_model(
    options: FitOptions,
    output: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    mapping_sha: str,
    slides: dict[str, Slide],
    labeled: set[str],
) -> Path:
    columns = FEATURE_COLUMNS[options.feature]
    feature = Feature(
        options.feature, tuple((options.key, column) for column in columns)
    )
    store = FeatureStore(cache_dir, manifest, "dev")
    if not store.has(feature):
        raise ValueError(f"Cache lacks development embeddings for {options.key}")
    eligible = store.eligible(feature)
    cohort = Cohort.from_slides(eligible & labeled, slides)
    if not cohort.ids:
        raise ValueError(f"No labeled development slides match {options.key}")
    records = len(set(cohort.groups))
    if options.alpha is None and records < options.folds:
        raise ValueError(
            f"Only {records} eligible records for {options.folds} folds. Lower --folds or set --alpha."
        )
    x = store.matrix(feature, cohort.ids)
    alphas = Protocol.from_config({}).alphas
    if options.alpha is None:
        cv = GroupKFold(n_splits=options.folds, shuffle=True, random_state=options.seed)
        alpha, weights, intercept = fit_ridge(
            x, cohort.y, tuple(cv.split(x, cohort.y, cohort.groups)), alphas
        )
        selection = {
            "method": "grouped_cv_mae",
            "folds": options.folds,
            "seed": options.seed,
            "alphas": alphas,
        }
    else:
        alpha = options.alpha
        coefficients, bias = ridge_path(x, cohort.y, (alpha,))
        weights, intercept = coefficients[:, 0], float(bias[0])
        selection = {"method": "fixed_alpha"}
    if not np.isfinite(weights).all() or not math.isfinite(intercept):
        raise ValueError("The ridge solution is not finite")
    if file_digest(options.mapping) != mapping_sha:
        raise ValueError("Data mapping changed while fitting")
    settings = manifest["settings"]
    result = {
        "schema_version": 1,
        "type": "ridge",
        "level": options.level,
        "mask": options.mask,
        "source_uri": manifest["sources"][options.key]["uri"],
        "feature": feature.name,
        "blocks": feature.blocks,
        "embedding_dim": settings["embedding_dim"],
        "layernorm_epsilon": settings["layernorm_epsilon"],
        "target_column": options.target_column,
        "coefficients": weights.tolist(),
        "intercept": intercept,
        "alpha": alpha,
        "alpha_selection": selection,
        "train_slides": len(cohort.ids),
        "train_records": records,
        "train_roster_digest": cohort.key,
        "missing_or_nonfinite_targets": len(slides) - len(labeled),
        "labeled_slides_without_features": len(labeled - eligible),
        "mapping_sha256": mapping_sha,
        "cache_digest": manifest["digest"],
    }
    path = output / "model.json"
    if path.exists():
        if signed_read(path)["digest"] != digest(result):
            raise ValueError(
                f"A different model is already saved at {path}. Choose a new --output."
            )
    else:
        signed_write(path, result)
    return path
