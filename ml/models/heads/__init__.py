"""Prediction heads ("hats")."""

from ml.models.heads.base import Head
from ml.models.heads.linear import LinearHead


__all__ = ["Head", "LinearHead"]
