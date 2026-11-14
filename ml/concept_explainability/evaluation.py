"""Fit the reused concept dictionary and evaluate native frozen-model effects."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from ml.concept_explainability.data import (
    corpus_identity,
    embedding_dataset,
    iter_slides,
    source_dataset,
)
from ml.concept_explainability.model import SourceRun, load_frozen_model, model_outputs
from ml.concept_explainability.upstream import (
    UPSTREAM_SHA,
    ConceptAutoencoder,
    code_share,
    fit_semi_nmf_online,
    gauge_fix_dictionary,
)


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class _FixtureDataset:
    """Small in-process fixture runner; production fits use upstream Ray batches."""

    def __init__(self, rows: np.ndarray):
        if len(rows) > 4096:
            raise ValueError(
                "In-process fixture fitting is limited to 4096 tiles; use Ray for a full fit."
            )
        self.rows = rows

    def select_columns(self, _columns):
        return self

    def iter_batches(
        self, batch_size, batch_format="numpy", local_shuffle_seed=None, **_kwargs
    ):
        indices = np.arange(len(self.rows))
        if local_shuffle_seed is not None:
            np.random.default_rng(local_shuffle_seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield {"embedding": self.rows[indices[start : start + batch_size]]}


def fit_dictionary(
    source: SourceRun, config: DictConfig, output_dir: Path, *, use_ray: bool = True
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Dictionary output directory must be empty; fitted dictionaries are immutable."
        )
    if int(config.n_components) != 16:
        raise ValueError("This replication fixes n_components=16.")
    dataset = source_dataset(source, "train")
    identity = corpus_identity(source, dataset)
    total = np.zeros(2560, dtype=np.float64)
    count = 0
    fixture_rows = []
    for slide in iter_slides(source, "train", dataset):
        total += slide.embeddings.astype(np.float64).sum(0)
        count += len(slide.embeddings)
        if not use_ray:
            if count > 4096:
                raise ValueError("Fixture fitting is limited to 4096 tiles.")
            fixture_rows.append(slide.embeddings)
    if count < 16:
        raise ValueError("At least 16 training tiles are required for the dictionary.")
    shift = (total / count).astype(np.float32)
    stream = (
        embedding_dataset(source, "train", dataset)
        if use_ray
        else _FixtureDataset(np.concatenate(fixture_rows))
    )
    h = gauge_fix_dictionary(
        fit_semi_nmf_online(stream, config, shift, np.ones_like(shift))
    )
    if h.shape != (16, 2560) or not np.isfinite(h).all():
        raise ValueError("Dictionary fit produced invalid atoms.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(h).rename_axis("component").to_parquet(output_dir / "h.parquet")
    np.save(output_dir / "shift.npy", shift)
    from hashlib import sha256

    dictionary_id = sha256(h.tobytes() + shift.tobytes()).hexdigest()
    save_json(
        output_dir / "manifest.json",
        {
            "dictionary_id": dictionary_id,
            "upstream_sha": UPSTREAM_SHA,
            "source_run_id": source.run_id,
            "training_sha": source.training_sha,
            "corpus": identity,
            "fit_split": "train",
            "n_tiles": count,
            "method": "semi_nmf",
            "center": True,
            "scale_power": 0.0,
            "metric_enabled": False,
            "config": OmegaConf.to_container(config, resolve=True),
            "fixture_only": not use_ray,
        },
    )
    autoencoder = ConceptAutoencoder(h, shift, False, int(config.nmf.nnls_iter))
    for slide in iter_slides(source, "train", dataset):
        write_codes(
            slide.keys,
            autoencoder.encode(slide.embeddings),
            output_dir / "coefficients" / "train" / f"{slide.slide_id}.parquet",
        )
    return output_dir


def write_codes(keys: pd.DataFrame, codes: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [
            keys.reset_index(drop=True),
            pd.DataFrame(codes, columns=[f"c{k}" for k in range(codes.shape[1])]),
        ],
        axis=1,
    ).to_parquet(path, index=False)


def load_dictionary(directory: Path) -> ConceptAutoencoder:
    from hashlib import sha256

    manifest = json.loads((directory / "manifest.json").read_text())
    h = pd.read_parquet(directory / "h.parquet").to_numpy(dtype=np.float32)
    shift = np.load(directory / "shift.npy", allow_pickle=False)
    if (
        h.shape != (16, 2560)
        or shift.shape != (2560,)
        or not np.isfinite(h).all()
        or not np.isfinite(shift).all()
    ):
        raise ValueError("Invalid dictionary dimensions or values.")
    if sha256(h.tobytes() + shift.tobytes()).hexdigest() != manifest["dictionary_id"]:
        raise ValueError("Dictionary checksum does not match its manifest.")
    return ConceptAutoencoder(
        h, shift, False, int(manifest["config"]["nmf"]["nnls_iter"])
    )


def encode_slide(slide, dictionary: ConceptAutoencoder) -> np.ndarray:
    return dictionary.encode(slide.embeddings)


def model_bag(module, slide, features: np.ndarray, device: str) -> torch.Tensor:
    from ml.models.aggregators.spatial_transformer import SpatialTransformerMIL

    bag = torch.as_tensor(features, dtype=torch.float32, device=device)
    if isinstance(module.aggregator, SpatialTransformerMIL):
        coordinates = torch.as_tensor(
            slide.keys[["x", "y"]].to_numpy(), dtype=torch.float32, device=device
        )
        bag = torch.cat([bag, coordinates], dim=1)
    return bag


def displayed_outputs(raw: np.ndarray, label_mode: str) -> np.ndarray:
    result = np.array(raw, dtype=np.float64, copy=True)
    if label_mode in {"type", "both"}:
        result[0] = torch.sigmoid(
            torch.tensor(float(raw[0]), dtype=torch.float64)
        ).item()
    return result


def concept_push(
    codes: np.ndarray,
    concept: int,
    spread: float,
    *,
    subtract: bool = False,
    step_sds: float = 0.25,
) -> np.ndarray:
    """Adam's existing two-sided coefficient intervention, with subtraction at zero."""
    pushed = codes.copy()
    step = step_sds * spread * codes.sum(axis=1)
    pushed[:, concept] = (
        np.maximum(codes[:, concept] - step, 0)
        if subtract
        else codes[:, concept] + step
    )
    return pushed


def evaluate_bag(
    module, slide, dictionary, spread, label_mode, device, *, codes=None, step_sds=0.25
):
    codes = encode_slide(slide, dictionary) if codes is None else codes

    def raw(features):
        return (
            model_outputs(
                module, model_bag(module, slide, features, device), label_mode
            )
            .cpu()
            .numpy()
        )

    original = raw(slide.embeddings)
    reconstructed = dictionary.decode(codes)
    reconstructed_raw = raw(reconstructed)
    base = displayed_outputs(reconstructed_raw, label_mode)
    added, removed, clipped = [], [], []
    for k in range(codes.shape[1]):
        plus = concept_push(codes, k, spread[k], step_sds=step_sds)
        added.append(displayed_outputs(raw(dictionary.decode(plus)), label_mode) - base)
        minus = concept_push(codes, k, spread[k], subtract=True, step_sds=step_sds)
        removed.append(
            base - displayed_outputs(raw(dictionary.decode(minus)), label_mode)
        )
        clipped.append(
            float(np.mean(codes[:, k] < step_sds * spread[k] * codes.sum(axis=1)))
        )
    return {
        "original": original,
        "reconstructed": reconstructed_raw,
        "add": np.asarray(added),
        "subtract": np.asarray(removed),
        "clipped": np.asarray(clipped),
        "embedding_squared_error": float(
            np.sum((slide.embeddings.astype(np.float64) - reconstructed) ** 2)
        ),
        "n_tiles": len(codes),
    }


def evaluate_concepts(
    source: SourceRun,
    dictionary_dir: Path,
    split: str,
    output_dir: Path,
    *,
    device: str = "cpu",
    step_sds: float = 0.25,
) -> Path:
    """Recompute the full trained model; never approximate nonlinear pooling."""
    if not np.isfinite(step_sds) or step_sds <= 0:
        raise ValueError("step_sds must be positive and finite.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Explanation output directory must be empty.")
    dictionary_manifest = json.loads((dictionary_dir / "manifest.json").read_text())
    identity = corpus_identity(source)
    if identity != dictionary_manifest["corpus"]:
        raise ValueError("Source training corpus does not match the fitted dictionary.")
    dictionary = load_dictionary(dictionary_dir)
    module = load_frozen_model(source).to(device)
    mode = str(source.config.label_mode)
    dataset = source_dataset(source, split)
    output_dir.mkdir(parents=True, exist_ok=True)
    means = []
    slide_ids = []
    for slide in iter_slides(source, split, dataset):
        codes = encode_slide(slide, dictionary)
        write_codes(
            slide.keys, codes, output_dir / "coefficients" / f"{slide.slide_id}.parquet"
        )
        means.append(code_share(codes).mean(0))
        slide_ids.append(slide.slide_id)
    means = np.asarray(means)
    sufficient = len(means) >= 2
    spread = means.std(axis=0, ddof=1) if sufficient else np.zeros(16)
    slide_rows, per_slide_effects = [], []
    squared_error = tiles = 0
    targets = ["luminal_a_probability"] if mode == "type" else ["mammaprint_index"]
    if mode == "both":
        targets = ["luminal_a_probability", "mammaprint_index"]
    for slide in iter_slides(source, split, dataset):
        frame = pd.read_parquet(
            output_dir / "coefficients" / f"{slide.slide_id}.parquet"
        )
        codes = frame[[f"c{k}" for k in range(16)]].to_numpy(dtype=np.float32)
        result = evaluate_bag(
            module,
            slide,
            dictionary,
            spread,
            mode,
            device,
            codes=codes,
            step_sds=step_sds,
        )
        squared_error += result["embedding_squared_error"]
        tiles += result["n_tiles"]
        for column, target in enumerate(targets):
            slide_rows.append(
                {
                    "slide_id": slide.slide_id,
                    "target": target,
                    "original_raw": float(result["original"][column]),
                    "reconstructed_raw": float(result["reconstructed"][column]),
                    "original": float(
                        displayed_outputs(result["original"], mode)[column]
                    ),
                    "reconstructed": float(
                        displayed_outputs(result["reconstructed"], mode)[column]
                    ),
                    "label": float(slide.label[column]),
                }
            )
            for k in range(16):
                per_slide_effects.append(
                    {
                        "slide_id": slide.slide_id,
                        "target": target,
                        "concept": k,
                        "add": float(result["add"][k, column]),
                        "subtract": float(result["subtract"][k, column]),
                        "clipped_tiles": float(result["clipped"][k]),
                    }
                )
    outputs = pd.DataFrame(slide_rows)
    effects = pd.DataFrame(per_slide_effects)
    summaries, presence, fidelity = [], [], []
    for target in targets:
        rows = outputs[outputs.target == target].set_index("slide_id").loc[slide_ids]
        classification = target == "luminal_a_probability"
        fidelity.append(
            {
                "target": target,
                "raw_rmse": float(
                    np.sqrt(np.mean((rows.original_raw - rows.reconstructed_raw) ** 2))
                ),
                "display_rmse": float(
                    np.sqrt(np.mean((rows.original - rows.reconstructed) ** 2))
                ),
                "label_agreement": float(
                    np.mean((rows.original_raw >= 0) == (rows.reconstructed_raw >= 0))
                )
                if classification
                else None,
                "n_slides": len(rows),
            }
        )
        groups = (
            {
                "A": rows.index[rows.original_raw >= 0],
                "B": rows.index[rows.original_raw < 0],
            }
            if classification
            else {"all": rows.index}
        )
        for group, ids in groups.items():
            if not len(ids):
                continue
            mask = np.isin(slide_ids, ids)
            for k in range(16):
                selected = effects[
                    (effects.target == target)
                    & (effects.concept == k)
                    & effects.slide_id.isin(ids)
                ]
                factor = (-100 if group == "B" else 100) if classification else 1
                added = factor * selected["add"].to_numpy()
                removed = factor * selected["subtract"].to_numpy()
                central = (added + removed) / 2
                summaries.append(
                    {
                        "target": target,
                        "group": group,
                        "concept": k,
                        "n_slides": len(ids),
                        "add": float(added.mean()),
                        "subtract": float(removed.mean()),
                        "effect": float(central.mean()),
                        "same_direction": float(
                            np.mean(np.sign(central) == np.sign(central.mean()))
                        ),
                        "disagrees": bool(
                            np.sign(added.mean()) != np.sign(removed.mean())
                        ),
                        "clipped_tiles": float(selected.clipped_tiles.mean()),
                        "units": "percentage_points" if classification else "index",
                        "status": "evaluated" if sufficient else "insufficient_cohort",
                    }
                )
                presence.append(
                    {
                        "target": target,
                        "group": group,
                        "concept": k,
                        "presence": float(
                            means[mask, k].mean() / max(means[:, k].mean(), 1e-12)
                        ),
                    }
                )
    outputs.to_parquet(output_dir / "slide_outputs.parquet", index=False)
    effects.to_parquet(output_dir / "slide_concept_effects.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(
        output_dir / "concept_effects.parquet", index=False
    )
    pd.DataFrame(presence).to_parquet(
        output_dir / "concept_presence.parquet", index=False
    )
    pd.DataFrame(fidelity).to_parquet(output_dir / "fidelity.parquet", index=False)
    save_json(
        output_dir / "manifest.json",
        {
            "dictionary_id": dictionary_manifest["dictionary_id"],
            "corpus": identity,
            "source_run_id": source.run_id,
            "training_sha": source.training_sha,
            "checkpoint_sha256": source.checkpoint_sha256,
            "split": split,
            "step_sds": step_sds,
            "spread_ddof": 1,
            "upstream_sha": UPSTREAM_SHA,
            "embedding_rmse": float(np.sqrt(squared_error / tiles)),
            "embedding_rmse_definition": "sqrt(sum squared feature errors / n_tiles)",
            "n_tiles": tiles,
            "n_slides": len(slide_ids),
            "effects_status": "evaluated" if sufficient else "insufficient_cohort",
            "fixture_only": dictionary_manifest["fixture_only"],
            "fidelity": fidelity,
        },
    )
    return output_dir
