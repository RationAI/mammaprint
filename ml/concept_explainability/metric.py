"""Native mean-linear classifier adaptation of Adam's deployed output metric.

Classifier algebra and hash sampling adapted from RationAI/ulcerative-colitis,
75c22a2a98d3b16a1c298735e2ea385cccfd07fb, explainability/model.py and
nmf_fit.py. See third_party/ibd-explainability-NOTICE.txt. The UC attention
term is inapplicable to native mean pooling; no artificial score is introduced.
"""

import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ml.concept_explainability.model import SourceRun, file_sha256, load_frozen_model
from ml.concept_explainability.upstream import iter_embedding_batches
from ml.models.aggregators import MeanPool
from ml.models.encoders.identity import IdentityEncoder
from ml.models.heads import LinearHead


METRIC_UPSTREAM_SHA = "75c22a2a98d3b16a1c298735e2ea385cccfd07fb"
METRIC_ADAPTER = "native_mean_linear_binary_classifier_v1"


def metric_enabled(config) -> bool:
    value = OmegaConf.select(config, "metric.enabled", default=False)
    if not isinstance(value, bool):
        raise ValueError("metric.enabled must be a boolean.")
    return value


def _validate_metric_data_config(config):
    """Validate raw single-scale bag semantics without instantiating any dataset."""
    for key in ("feature_dim", "ml.encoder.out_dim", "ml.aggregator.feature_dim", "ml.head.in_dim"):
        width = OmegaConf.select(config, key)
        if isinstance(width, bool) or not isinstance(width, Integral) or width != 2560:
            raise ValueError(f"Deployed metric requires {key}=2560.")
    levels = OmegaConf.select(config, "dataset.levels")
    if not OmegaConf.is_dict(levels) or len(levels) != 1:
        raise ValueError("Deployed metric requires exactly one dataset level.")
    level, card = next(iter(levels.items()))
    targets = {"ml.data.SingleScaleDataset", "ml.data.datasets.single_scale.SingleScaleDataset"}
    for split in ("train", "val", "test"):
        dataset = OmegaConf.select(config, f"datamodule.{split}")
        if dataset is None and split != "train":
            continue
        if not OmegaConf.is_dict(dataset) or dataset.get("_target_") not in targets:
            raise ValueError(f"Deployed metric requires exact SingleScaleDataset for {split}.")
        if dataset.get("split") != split or dataset.get("label_mode", "type") != "type":
            raise ValueError(f"Deployed metric requires matching split and type labels for {split}.")
        selected = dataset.get("levels")
        if not OmegaConf.is_dict(selected) or len(selected) != 1 or str(next(iter(selected))) != str(level):
            raise ValueError(f"Deployed metric requires one matching dataset level for {split}.")
        selected_card = next(iter(selected.values()))
        # iter_slides/corpus_identity use the global card to label the actual bag.
        for field in ("mpp", "tile_extent", "stride", f"uris.{split}"):
            expected = OmegaConf.select(card, field, default=card.get("tile_extent") if field == "stride" else None)
            actual = OmegaConf.select(selected_card, field, default=selected_card.get("tile_extent") if field == "stride" else None)
            if expected is None or actual != expected:
                raise ValueError(f"Deployed metric {split} dataset {field} differs from its source card.")


def preflight_metric_source(source: SourceRun):
    """Validate the metric graph and return its frozen CPU model before data reads."""
    allowed = {
        "encoder": {"ml.models.encoders.IdentityEncoder", "ml.models.encoders.identity.IdentityEncoder"},
        "aggregator": {"ml.models.aggregators.MeanPool", "ml.models.aggregators.mean.MeanPool"},
        "head": {"ml.models.heads.LinearHead", "ml.models.heads.linear.LinearHead"},
        "loss": {"torch.nn.BCEWithLogitsLoss", "torch.nn.modules.loss.BCEWithLogitsLoss"},
    }
    if source.config.get("label_mode") != "type":
        raise ValueError("Deployed metric requires label_mode=type; index/joint are unsupported.")
    for name, targets in allowed.items():
        target = OmegaConf.select(source.config, f"ml.{name}._target_")
        if target not in targets:
            raise ValueError(f"Deployed metric does not support {name}={target}; requires IdentityEncoder + MeanPool + LinearHead + BCE.")
    activation = OmegaConf.select(source.config, "ml.output_activation._target_")
    if activation not in (None, "torch.nn.Identity", "torch.nn.modules.linear.Identity"):
        raise ValueError("Deployed metric requires Identity output activation.")
    _validate_metric_data_config(source.config)
    module = load_frozen_model(source)
    if (
        type(module.encoder) is not IdentityEncoder
        or type(module.aggregator) is not MeanPool
        or type(module.head) is not LinearHead
        or type(module.loss) is not torch.nn.BCEWithLogitsLoss
        or module.head.linear.out_features != 1
        or module.encoder.out_dim != 2560
        or module.encoder.out_dim != module.head.linear.in_features
        or module.aggregator.out_dim != module.encoder.out_dim
    ):
        raise ValueError("Deployed metric requires the native mean-linear single BCE logit graph.")
    source_binding(source)  # Verify checkpoint bytes as well as architecture.
    return module


def source_binding(source: SourceRun) -> dict:
    """Checkpoint and complete resolved-config binding for model-specific dictionaries."""
    if source.checkpoint_path is None or source.checkpoint_sha256 is None:
        raise ValueError("Deployed metric requires a checkpoint and checksum.")
    if file_sha256(source.checkpoint_path) != source.checkpoint_sha256:
        raise ValueError("Deployed metric checkpoint checksum does not match source bytes.")
    config = OmegaConf.to_container(source.config, resolve=True, throw_on_missing=True)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "source_run_id": source.run_id,
        "training_sha": source.training_sha,
        "checkpoint_sha256": source.checkpoint_sha256,
        "resolved_config_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def classifier_output_metric(z, weight, bias, *, ridge=0.3) -> np.ndarray:
    """UC classifier component with class-centered binary logits [0, z].

    Keeps upstream float32 classifier arithmetic and float64 sample/matrix
    arithmetic, trace normalization, ridge, CPU eigh and float32 T orientation.
    """
    z = np.asarray(z)
    weight, bias = np.asarray(weight), np.asarray(bias)
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("Metric sample must have at least two finite tile rows.")
    if weight.shape != (1, z.shape[1]) or bias.shape != (1,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Metric classifier requires finite single-logit weights and bias.")
    if isinstance(ridge, bool) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Metric ridge must be positive and finite.")
    z = z.astype(np.float64)
    theta = np.concatenate([np.zeros_like(weight), weight])
    b = np.concatenate([np.zeros_like(bias), bias])
    theta, b = theta - theta.mean(axis=0), b - b.mean()
    logits = z @ theta.T + b
    dim = z.shape[1]
    metric = np.zeros((dim, dim))
    for c in range(len(theta)):
        variance = logits[:, c].var()
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Metric classifier output variance must be finite and nonzero.")
        metric += np.outer(theta[c], theta[c]) / variance
    trace = np.trace(metric)
    if not np.isfinite(metric).all() or not np.isfinite(trace) or trace <= 0:
        raise ValueError("Metric classifier variance produced an invalid metric.")
    metric *= dim / trace
    metric += ridge * np.eye(dim)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("Metric must be positive definite.")
    return (eigenvectors * np.sqrt(eigenvalues)).T.astype(np.float32)


def tile_uniform(metadata: pd.DataFrame, random_state: int) -> np.ndarray:
    """Exact deployed seeded hash-threshold sampling, including pandas dtypes."""
    hashes = pd.util.hash_pandas_object(
        metadata[["slide_id", "x", "y"]], index=False, hash_key=f"{random_state:016d}"[-16:]
    )
    return hashes.to_numpy() / 2.0**64


def sample_metric_rows(dataset, n_rows, shift, *, batch_size, n_tiles=20_000, seed=42):
    """Return shifted float32 sample and train-key provenance; expected size, no cap."""
    if isinstance(n_tiles, bool) or not isinstance(n_tiles, Integral) or n_tiles <= 0 or n_rows <= 0:
        raise ValueError("Metric sample size and population must be positive integers.")
    fraction = n_tiles / n_rows
    chunks = []
    digest = hashlib.sha256()
    seen = 0
    for rows, metadata in iter_embedding_batches(
        dataset, batch_size, shift, np.ones_like(shift),
        metadata_columns=("slide_id", "x", "y"), clip=False,
    ):
        seen += len(rows)
        mask = tile_uniform(metadata, seed) < fraction
        chunks.append(rows[mask])
        for row in metadata.loc[mask].itertuples(index=False):
            digest.update((json.dumps([str(row.slide_id), int(row.x), int(row.y)], separators=(",", ":")) + "\n").encode())
    if seen != n_rows:
        raise ValueError("Metric sample stream row count differs from training population.")
    sample = np.concatenate(chunks) if chunks else np.empty((0, len(shift)), np.float32)
    if not len(sample):
        raise ValueError("Metric hash sample is empty; increase metric.n_tiles.")
    return sample, {
        "split": "train", "requested_tiles": int(n_tiles), "actual_tiles": len(sample),
        "population_tiles": int(n_rows), "seed": int(seed),
        "sampling": "pandas_hash_threshold_v1", "pandas_version": pd.__version__,
        "selected_keys_sha256": digest.hexdigest(),
    }


def metric_dictionary_id(h, shift, transform, binding, adapter=METRIC_ADAPTER):
    """Versioned identity; legacy plain IDs deliberately use their original formula."""
    digest = hashlib.sha256(b"mammaprint-metric-dictionary-v2\0")
    digest.update(json.dumps({"adapter": adapter, "binding": binding}, sort_keys=True, separators=(",", ":")).encode())
    for array in (h, shift, transform):
        digest.update(json.dumps({"shape": array.shape, "dtype": str(array.dtype)}, sort_keys=True).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()
