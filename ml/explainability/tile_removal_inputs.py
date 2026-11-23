"""Resolve a pinned checkpoint and its validation-only inputs."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor, nn

from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.mean import MeanPool
from ml.models.encoders.identity import IdentityEncoder
from ml.models.encoders.tile_projector import TileProjector
from ml.models.heads.linear import LinearHead
from ml.models.module import MammaprintModule


_SHA256 = re.compile(r"[a-f0-9]{64}")
_RUN_ID = re.compile(r"[a-f0-9]{32}")
_SOURCE_REVISION = re.compile(r"[a-f0-9]{40}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(location: str, root: Path) -> Path:
    return root / location


def fetch(location: str, root: Path, cache: Path) -> Path:
    scheme = urlsplit(location).scheme
    if not scheme:
        return local_path(location, root)
    if scheme == "file":
        from urllib.parse import unquote

        return Path(unquote(urlsplit(location).path))
    if scheme not in {"mlflow-artifacts", "runs"}:
        raise ValueError(f"Unsupported input artifact scheme {scheme!r}.")
    from mlflow.artifacts import download_artifacts

    cache.mkdir(parents=True, exist_ok=True)
    return Path(download_artifacts(artifact_uri=location, dst_path=str(cache)))


def pinned_file(
    location: str, expected: str, root: Path, cache: Path, name: str
) -> Path:
    if not _SHA256.fullmatch(expected):
        raise ValueError(f"A lowercase SHA-256 is required for {name}.")
    path = fetch(location, root, cache / name)
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{name} digest mismatch.")
    return path


def verify_model_spec(spec: dict[str, Any]) -> None:
    for key in ("model_id", "source_run_id", "source_revision"):
        if not isinstance(spec.get(key), str):
            raise ValueError(f"Model specification requires {key}.")
    if not _SAFE_NAME.fullmatch(spec["model_id"]):
        raise ValueError("Invalid model ID.")
    if not _RUN_ID.fullmatch(spec["source_run_id"]):
        raise ValueError("Pin the source training run by exact MLflow run ID.")
    if not _SOURCE_REVISION.fullmatch(spec["source_revision"]):
        raise ValueError("Pin the source training revision by full Git SHA.")
    for key in ("checkpoint", "config", "label_mapping", "validation_source"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError(f"Model specification requires {key}.")
    for key in ("checkpoint", "config", "label_mapping"):
        if not _SHA256.fullmatch(str(spec.get(f"{key}_sha256", ""))):
            raise ValueError(f"Pin {key} by SHA-256.")
    for key in ("checkpoint", "config"):
        location = urlsplit(spec[key])
        if location.scheme == "runs":
            run_id = location.path.lstrip("/").split("/")[0]
        elif location.scheme == "mlflow-artifacts":
            parts = location.path.strip("/").split("/")
            run_id = parts[1] if len(parts) > 2 and parts[2] == "artifacts" else ""
        else:
            continue
        if run_id != spec["source_run_id"]:
            raise ValueError(f"{key} artifact belongs to a different training run.")


def _class(label: str) -> str | None:
    return {"a luminal": "A", "b luminal": "B"}.get(label)


def _number(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class Slide:
    slide_id: str
    wsi: str
    recorded_class: str | None
    recorded_index: float | None
    patient_id: str | None


def load_roster(path: Path, patient_column: str | None) -> tuple[list[Slide], set[str]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"path", "split", "type", "mammaprint_index"}
    if patient_column is not None:
        required.add(patient_column)
    if missing := required - set(frame):
        raise ValueError(f"Mapping is missing columns {sorted(missing)}.")
    names = frame["path"].map(lambda p: Path(p).stem)
    if names.duplicated().any() or any(not name for name in names):
        raise ValueError("Slide mapping has missing or ambiguous technical slide IDs.")
    validation = frame["split"] == "val"
    if not validation.any():
        raise ValueError("Slide mapping has no original validation assignments.")
    rows = []
    for pos, slide_id in names[validation].items():
        row = frame.loc[pos]
        rows.append(
            Slide(
                slide_id,
                row["path"],
                _class(row["type"]),
                _number(row["mammaprint_index"]),
                (row[patient_column] or None) if patient_column else None,
            )
        )
    return sorted(rows, key=lambda row: row.slide_id), set(names[~validation])


class ValidationBags:
    def __init__(self, source: str, root: Path, cache: Path) -> None:
        self.source = source.rstrip("/")
        self.root = root
        self.cache = cache
        self.local: Path | None = None
        if urlsplit(source).scheme in {"", "file"}:
            self.local = fetch(source, root, cache)
            if not self.local.is_dir():
                raise FileNotFoundError(f"Validation artifact is missing: {self.local}")
            paths = [path.name for path in self.local.glob("*.parquet")]
        else:
            from mlflow.artifacts import list_artifacts

            if urlsplit(source).scheme not in {"mlflow-artifacts", "runs"}:
                raise ValueError(
                    "Validation source must be a local or MLflow artifact."
                )
            paths = [
                Path(item.path).name
                for item in list_artifacts(artifact_uri=self.source)
                if not item.is_dir and item.path.endswith(".parquet")
            ]
        if len(set(paths)) != len(paths):
            raise ValueError("Validation artifact has ambiguous bag filenames.")
        self.names = {Path(path).stem for path in paths}

    def bag_path(self, slide_id: str) -> Path | None:
        if slide_id not in self.names:
            return None
        filename = f"{slide_id}.parquet"
        if self.local is not None:
            return self.local / filename
        return fetch(f"{self.source}/{quote(filename)}", self.root, self.cache)


@dataclass
class PreparedModel:
    model: MammaprintModule
    bags: ValidationBags
    roster: list[Slide]
    provenance: dict[str, Any]
    predictions: dict[str, float]


def prepare_model(
    spec: dict[str, Any], root: Path, cache: Path, device: torch.device
) -> PreparedModel:
    checkpoint = pinned_file(
        spec["checkpoint"], spec["checkpoint_sha256"], root, cache, "checkpoint"
    )
    config_path = pinned_file(
        spec["config"], spec["config_sha256"], root, cache, "config"
    )
    labels_path = pinned_file(
        spec["label_mapping"],
        spec["label_mapping_sha256"],
        root,
        cache,
        "label_mapping",
    )
    config = OmegaConf.load(config_path)
    validation = config.datamodule.val
    if (
        config.label_mode != "index"
        or validation.split != "val"
        or validation.label_mode != "index"
        or validation._target_
        not in {
            "ml.data.SingleScaleDataset",
            "ml.data.datasets.single_scale.SingleScaleDataset",
        }
        or len(validation.levels) != 1
        or "tile_cap" in validation
    ):
        raise ValueError(
            "Expected an uncapped single-scale validation regression dataset."
        )
    if str(validation.data_mapping) != spec["label_mapping"]:
        raise ValueError("The label mapping differs from the frozen validation config.")
    level, card = next(iter(validation.levels.items()))
    if card.uris.val != spec["validation_source"]:
        raise ValueError("Validation artifact differs from the frozen training config.")
    if spec["validation_source"] in (card.uris.get("train"), card.uris.get("test")):
        raise ValueError("The validation artifact is shared with another split.")
    if (
        int(level) != int(card.level)
        or min(card.tile_extent, card.stride) <= 0
        or not (math.isfinite(card.mpp) and card.mpp > 0)
    ):
        raise ValueError("Invalid frozen tiling configuration.")

    model_config = config.ml
    encoder = instantiate(model_config.encoder)
    aggregator = instantiate(model_config.aggregator)
    head = instantiate(model_config.head)
    activation = instantiate(model_config.get("output_activation")) or nn.Identity()
    if (
        model_config._target_
        not in {"ml.models.MammaprintModule", "ml.models.module.MammaprintModule"}
        or type(encoder) not in {IdentityEncoder, TileProjector}
        or type(aggregator) not in {MeanPool, AttentionMIL}
        or type(head) is not LinearHead
        or head.out_dim != 1
        or type(activation) is not nn.Identity
    ):
        raise ValueError("Unsupported frozen encoder, pool, head or activation.")
    model = MammaprintModule(encoder, aggregator, head, activation)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval().requires_grad_(False)

    roster, other_splits = load_roster(labels_path, spec.get("patient_column"))
    bags = ValidationBags(spec["validation_source"], root, cache / "bags")
    if overlap := bags.names & other_splits:
        raise ValueError(
            f"Validation artifact contains {len(overlap)} non-validation slides."
        )
    predictions: dict[str, float] = {}
    if source := spec.get("validation_predictions"):
        saved = pinned_file(
            source["path"], source["sha256"], root, cache, "validation_predictions"
        )
        records = pd.read_csv(saved, dtype={"slide_id": str})
        if (
            records["slide_id"].duplicated().any()
            or not set(records["slide_id"]).issubset({row.slide_id for row in roster})
            or not np.isfinite(records["prediction"].to_numpy(dtype=float)).all()
        ):
            raise ValueError(
                "Saved validation predictions failed split or value checks."
            )
        predictions = dict(zip(records["slide_id"], records["prediction"], strict=True))

    return PreparedModel(
        model,
        bags,
        roster,
        {
            **spec,
            "split": "val",
            "aggregator": type(aggregator).__name__,
            "level": int(level),
            "tile_extent": int(card.tile_extent),
            "stride": int(card.stride),
            "configured_mpp": float(card.mpp),
            "configured_label_mapping": str(validation.data_mapping),
        },
        predictions,
    )


@torch.inference_mode()
def tile_statistics(
    model: MammaprintModule, bag: Tensor
) -> tuple[np.ndarray, np.ndarray]:
    features = model.encoder(bag)
    if isinstance(model.aggregator, AttentionMIL):
        features, logits = model.aggregator.attention_inputs(features)
    else:
        logits = torch.zeros(len(features), device=features.device)
    head = model.head
    if not isinstance(head, LinearHead) or head.linear.bias is None:
        raise ValueError("Tile statistics require an affine scalar head.")
    weight = head.linear.weight.detach().double()
    bias = head.linear.bias.detach().double()
    scores = (features.double() @ weight.T + bias).squeeze(1)
    return scores.cpu().numpy(), logits.double().cpu().numpy()
