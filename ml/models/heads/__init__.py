"""Prediction heads ("hats")."""

from ml.models.heads.base import Head
from ml.models.heads.linear import LinearHead
from ml.models.heads.mlp import MLPHead


__all__ = ["Head", "LinearHead", "MLPHead"]
