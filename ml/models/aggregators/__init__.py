"""Bag aggregators (MIL pooling): mean, max, gated attention, transformer."""

# ── Single-level aggregators (flat Bag) ──────────────────────────────────────
from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.base import Aggregator
from ml.models.aggregators.max import MaxPool
from ml.models.aggregators.mean import MeanPool
from ml.models.aggregators.spatial_transformer import SpatialTransformerMIL
from ml.models.aggregators.transformer import TransformerMIL


# ── Multilevel (comment out to ship single-level only) ───────────────────────
# from ml.models.aggregators.multiscale import MultiScaleMIL


__all__ = [
    "Aggregator",
    "AttentionMIL",
    "MaxPool",
    "MeanPool",
    "SpatialTransformerMIL",
    "TransformerMIL",
    # ── Multilevel ──
    # "MultiScaleMIL",
]
