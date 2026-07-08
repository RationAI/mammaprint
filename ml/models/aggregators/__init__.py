"""Bag aggregators (MIL pooling): mean, max, gated attention, transformer."""

# from ml.models.aggregators.attention import AttentionMIL
from ml.models.aggregators.base import Aggregator
# from ml.models.aggregators.max import MaxPool
from ml.models.aggregators.mean import MeanPool
# from ml.models.aggregators.multiscale import MultiScaleMIL
# from ml.models.aggregators.transformer import TransformerMIL


__all__ = [
    "Aggregator",
    # "AttentionMIL",
    # "MaxPool",
    "MeanPool",
    # "MultiScaleMIL",
    # "TransformerMIL",
]
