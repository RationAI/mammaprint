"""Encoder abstract base class.

An encoder maps a bag of raw tiles to a bag of per-tile feature vectors. On the
precomputed-embeddings path the encoder is :class:`~ml.models.encoders.identity.IdentityEncoder`
(the features already exist); on the end-to-end path it is an image backbone such
as :class:`~ml.models.encoders.vgg16.VGG16Encoder` applied to every tile.

Concrete encoders live in sibling modules, one implementation per file, and are
selected via the ``ml/encoder`` Hydra config group. The generic
:class:`~ml.models.module.MammaprintModule` only depends on this contract, never
on a concrete class.
"""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Encoder(nn.Module, ABC):
    """Maps a bag of tiles ``(N, ...)`` to per-tile features ``(N, out_dim)``.

    Implementations must set :attr:`out_dim` so the downstream aggregator and head
    can be wired without hard-coding feature sizes.
    """

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Dimensionality of a single per-tile feature vector."""

    @abstractmethod
    def forward(self, tiles: Tensor) -> Tensor:
        """Encode a bag of tiles.

        Args:
            tiles: A bag for one slide. Shape ``(N, C, H, W)`` for image
                backbones or ``(N, D)`` for the identity encoder, where ``N`` is
                the number of tiles.

        Returns:
            Per-tile features of shape ``(N, out_dim)``.
        """


__all__ = ["Encoder"]
