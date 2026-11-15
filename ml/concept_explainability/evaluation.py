"""Fit the reused concept dictionary and evaluate native frozen-model effects."""

import json
from numbers import Integral
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
from ml.concept_explainability.metric import (
    METRIC_ADAPTER,
    METRIC_UPSTREAM_SHA,
    classifier_output_metric,
    metric_dictionary_id,
    metric_enabled,
    preflight_metric_source,
    sample_metric_rows,
    source_binding,
)
from ml.concept_explainability.model import (
    SourceRun,
    file_sha256,
    load_frozen_model,
    model_outputs,
)
from ml.concept_explainability.upstream import (
    UPSTREAM_SHA,
    ConceptAutoencoder,
    code_share,
    fit_semi_nmf_online,
    gauge_fix_dictionary,
)


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_n_components(value: Any) -> int:
    """Validate K without silently truncating or coercing configuration values."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("n_components must be a positive integer.")
    return int(value)


def dictionary_n_components(manifest: dict[str, Any]) -> int:
    """Read K from new manifests or the saved fit config in older dictionaries."""
    configured = validate_n_components(manifest["config"]["n_components"])
    declared = validate_n_components(manifest.get("n_components", configured))
    if declared != configured:
        raise ValueError("Dictionary n_components does not match its saved fit config.")
    return declared


def validate_codes(codes: np.ndarray, n_components: int, n_rows: int) -> np.ndarray:
    if (
        codes.shape != (n_rows, n_components)
        or not np.isfinite(codes).all()
        or np.any(codes < 0)
    ):
        raise ValueError(
            "Coefficient dimensions or values do not match the dictionary."
        )
    return codes


def read_codes(frame: pd.DataFrame, n_components: int) -> np.ndarray:
    expected = [f"c{k}" for k in range(n_components)]
    actual = [
        c for c in frame if isinstance(c, str) and c.startswith("c") and c[1:].isdigit()
    ]
    if actual != expected:
        raise ValueError("Coefficient columns do not match dictionary n_components.")
    return validate_codes(
        frame[expected].to_numpy(dtype=np.float32), n_components, len(frame)
    )


class _FixtureDataset:
    """Small in-process fixture runner; production fits use upstream Ray batches."""

    def __init__(self, rows: np.ndarray, keys: pd.DataFrame | None = None):
        if len(rows) > 4096:
            raise ValueError(
                "In-process fixture fitting is limited to 4096 tiles; use Ray for a full fit."
            )
        self.rows = rows
        self.keys = keys

    def select_columns(self, _columns):
        return self

    def iter_batches(
        self, batch_size, batch_format="numpy", local_shuffle_seed=None, **_kwargs
    ):
        indices = np.arange(len(self.rows))
        if local_shuffle_seed is not None:
            np.random.default_rng(local_shuffle_seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = {"embedding": self.rows[selected]}
            if self.keys is not None:
                batch.update({column: self.keys[column].to_numpy()[selected] for column in self.keys})
            yield batch


def fit_dictionary(
    source: SourceRun, config: DictConfig, output_dir: Path, *, use_ray: bool = True
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Dictionary output directory must be empty; fitted dictionaries are immutable."
        )
    n_components = validate_n_components(config.n_components)
    enabled = metric_enabled(config)
    module = preflight_metric_source(source) if enabled else None
    dataset = source_dataset(source, "train")
    identity = corpus_identity(source, dataset)
    total = np.zeros(2560, dtype=np.float64)
    count = 0
    fixture_rows = []
    fixture_keys = []
    for slide in iter_slides(source, "train", dataset):
        total += slide.embeddings.astype(np.float64).sum(0)
        count += len(slide.embeddings)
        if not use_ray:
            if count > 4096:
                raise ValueError("Fixture fitting is limited to 4096 tiles.")
            fixture_rows.append(slide.embeddings)
            if enabled:
                fixture_keys.append(slide.keys[["slide_id", "x", "y"]])
    if count < n_components:
        raise ValueError(
            f"At least {n_components} training tiles are required for the dictionary."
        )
    # The unchanged upstream initializer samples at most 20,000 first-batch rows.
    if min(count, config.nmf.batch_size, 20_000) < n_components:
        raise ValueError(
            f"Dictionary initialization requires at least {n_components} sampled first-batch rows."
        )
    shift = (total / count).astype(np.float32)
    stream = (
        embedding_dataset(source, "train", dataset, include_keys=enabled)
        if use_ray
        else _FixtureDataset(np.concatenate(fixture_rows), pd.concat(fixture_keys, ignore_index=True) if enabled else None)
    )
    transform = None
    metric_metadata = None
    if enabled:
        sample, sample_metadata = sample_metric_rows(
            stream, count, shift, batch_size=config.nmf.batch_size,
            n_tiles=config.metric.get("n_tiles", 20_000), seed=config.nmf.random_state,
        )
        # Preserve upstream float32 subtract/restore roundoff before metric estimation.
        transform = classifier_output_metric(
            sample + shift, module.head.linear.weight.detach().cpu().numpy(),
            module.head.linear.bias.detach().cpu().numpy(), ridge=config.metric.get("ridge", 0.3),
        )
        requested_attention = config.metric.get("attention_weight", 1.0)
        if isinstance(requested_attention, bool) or not np.isfinite(requested_attention) or requested_attention < 0:
            raise ValueError("metric.attention_weight must be finite and nonnegative.")
        metric_metadata = {
            "adapter": METRIC_ADAPTER, "upstream_sha": METRIC_UPSTREAM_SHA,
            "binding": source_binding(source), "sample": sample_metadata,
            "ridge": float(config.metric.get("ridge", 0.3)),
            "attention_weight_requested": float(requested_attention),
            "attention_applicable": False, "attention_terms": 0,
            "attention_reason": "aggregator_has_no_attention", "output_terms": 2,
            "binary_logits": "[0,z], centered across classes",
        }
    if enabled:
        h = fit_semi_nmf_online(stream, config, shift, np.ones_like(shift), transform=transform)
        h = (h @ np.linalg.inv(transform).T).astype(np.float32)
    else:
        h = fit_semi_nmf_online(stream, config, shift, np.ones_like(shift))
    # Inverse mapping must precede the raw-space gauge fix.
    h = gauge_fix_dictionary(h)
    if h.shape != (n_components, 2560) or not np.isfinite(h).all():
        raise ValueError("Dictionary fit produced invalid atoms.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(h).rename_axis("component").to_parquet(output_dir / "h.parquet")
    np.save(output_dir / "shift.npy", shift)
    from hashlib import sha256

    if enabled:
        np.save(output_dir / "transform.npy", transform)
        metric_metadata["transform"] = {
            "file": "transform.npy", "shape": list(transform.shape),
            "dtype": str(transform.dtype), "sha256": file_sha256(output_dir / "transform.npy"),
        }
    dictionary_id = (
        metric_dictionary_id(h, shift, transform, metric_metadata["binding"])
        if enabled else sha256(h.tobytes() + shift.tobytes()).hexdigest()
    )
    save_json(
        output_dir / "manifest.json",
        {
            "dictionary_id": dictionary_id,
            "n_components": n_components,
            "upstream_sha": UPSTREAM_SHA,
            "source_run_id": source.run_id,
            "training_sha": source.training_sha,
            "corpus": identity,
            "fit_split": "train",
            "n_tiles": count,
            "method": "semi_nmf",
            "center": True,
            "scale_power": 0.0,
            "metric_enabled": enabled,
            **({"format_version": 2, "metric": metric_metadata} if enabled else {}),
            "config": OmegaConf.to_container(config, resolve=True),
            "fixture_only": not use_ray,
        },
    )
    autoencoder = ConceptAutoencoder(h, shift, False, int(config.nmf.nnls_iter), transform=transform)
    for slide in iter_slides(source, "train", dataset):
        write_codes(
            slide.keys,
            autoencoder.encode(slide.embeddings),
            output_dir / "coefficients" / "train" / f"{slide.slide_id}.parquet",
        )
    return output_dir


def write_codes(keys: pd.DataFrame, codes: np.ndarray, path: Path) -> None:
    if codes.ndim != 2:
        raise ValueError("Coefficients must be a two-dimensional matrix.")
    validate_codes(codes, validate_n_components(codes.shape[1]), len(keys))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [
            keys.reset_index(drop=True),
            pd.DataFrame(codes, columns=[f"c{k}" for k in range(codes.shape[1])]),
        ],
        axis=1,
    ).to_parquet(path, index=False)


def load_dictionary(directory: Path, *, source: SourceRun | None = None) -> ConceptAutoencoder:
    from hashlib import sha256

    manifest = json.loads((directory / "manifest.json").read_text())
    n_components = dictionary_n_components(manifest)
    h = pd.read_parquet(directory / "h.parquet").to_numpy(dtype=np.float32)
    shift = np.load(directory / "shift.npy", allow_pickle=False)
    if (
        h.shape != (n_components, 2560)
        or shift.shape != (2560,)
        or not np.isfinite(h).all()
        or not np.isfinite(shift).all()
    ):
        raise ValueError("Invalid dictionary dimensions or values.")
    enabled = manifest.get("metric_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("Dictionary metric_enabled must be a boolean.")
    transform = None
    if enabled:
        metadata = manifest.get("metric", {})
        if manifest.get("format_version") != 2 or metadata.get("adapter") != METRIC_ADAPTER:
            raise ValueError("Unsupported metric dictionary format or adapter.")
        saved_transform = metadata.get("transform", {})
        path = directory / "transform.npy"
        if not path.is_file() or saved_transform.get("file") != "transform.npy":
            raise ValueError("Metric dictionary transform is missing.")
        if file_sha256(path) != saved_transform.get("sha256"):
            raise ValueError("Metric transform checksum does not match its manifest.")
        try:
            transform = np.load(path, allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise ValueError("Invalid metric transform file.") from exc
        if (transform.shape != (2560, 2560) or transform.dtype != np.float32
            or saved_transform.get("shape") != list(transform.shape)
            or saved_transform.get("dtype") != str(transform.dtype)
            or not np.isfinite(transform).all()):
            raise ValueError("Invalid metric transform shape, dtype or values.")
        sign, logdet = np.linalg.slogdet(transform.astype(np.float64))
        if sign == 0 or not np.isfinite(logdet):
            raise ValueError("Metric transform must be invertible.")
        binding = metadata.get("binding")
        if not isinstance(binding, dict) or not {"source_run_id", "training_sha", "checkpoint_sha256", "resolved_config_sha256"}.issubset(binding):
            raise ValueError("Metric dictionary source binding is missing.")
        expected_id = metric_dictionary_id(h, shift, transform, binding, metadata["adapter"])
        if source is not None and source_binding(source) != binding:
            raise ValueError("Metric dictionary source binding does not match this model/config.")
    else:
        if manifest.get("format_version", 1) >= 2 and ("metric" in manifest or (directory / "transform.npy").exists()):
            raise ValueError("Contradictory metric flags or transform artifact.")
        expected_id = sha256(h.tobytes() + shift.tobytes()).hexdigest()
    if expected_id != manifest["dictionary_id"]:
        raise ValueError("Dictionary checksum does not match its manifest.")
    return ConceptAutoencoder(
        h, shift, False, int(manifest["config"]["nmf"]["nnls_iter"]), transform=transform
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
    n_components = validate_n_components(dictionary.dictionary.shape[0])
    codes = encode_slide(slide, dictionary) if codes is None else codes
    validate_codes(codes, n_components, len(slide.embeddings))
    spread = np.asarray(spread)
    if (
        spread.shape != (n_components,)
        or not np.isfinite(spread).all()
        or np.any(spread < 0)
    ):
        raise ValueError(
            "Intervention spread must match dictionary n_components and be finite and nonnegative."
        )

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
    dictionary = load_dictionary(dictionary_dir, source=source)
    identity = corpus_identity(source)
    if identity != dictionary_manifest["corpus"]:
        raise ValueError("Source training corpus does not match the fitted dictionary.")
    n_components = dictionary.dictionary.shape[0]
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
    spread = means.std(axis=0, ddof=1) if sufficient else np.zeros(n_components)
    slide_rows, per_slide_effects = [], []
    squared_error = tiles = 0
    targets = ["luminal_a_probability"] if mode == "type" else ["mammaprint_index"]
    if mode == "both":
        targets = ["luminal_a_probability", "mammaprint_index"]
    for slide in iter_slides(source, split, dataset):
        frame = pd.read_parquet(
            output_dir / "coefficients" / f"{slide.slide_id}.parquet"
        )
        codes = read_codes(frame, n_components)
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
            for k in range(n_components):
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
            for k in range(n_components):
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
            "metric_enabled": dictionary_manifest.get("metric_enabled", False),
            **({"metric": dictionary_manifest["metric"]} if dictionary_manifest.get("metric_enabled") else {}),
            "n_components": n_components,
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
