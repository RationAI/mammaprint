"""VGG16 image-backbone encoder for the end-to-end (raw-tile) path."""

import torch
from torch import Tensor, nn
from torchvision.models import VGG16_Weights, vgg16

from ml.models.encoders.base import Encoder


class VGG16Encoder(Encoder):
    """Encodes raw tiles with VGG16 convolutional features + global max pool.

    Applies the ``torchvision`` VGG16 ``features`` stack to every tile in a bag
    and reduces each tile's feature map to a single ``512``-dim vector via
    adaptive global max pooling. The classifier/fully-connected part of VGG16 is
    dropped — the slide-level prediction is produced downstream by the
    aggregator + head.

    Args:
        pretrained: Load ImageNet-pretrained weights when ``True``.
        freeze: Freeze the backbone (no gradient updates) when ``True`` — useful
            for training only the aggregator/head on top of a fixed backbone.
    """

    _FEATURE_DIM = 512

    def __init__(self, pretrained: bool = True, freeze: bool = False) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = vgg16(weights=weights).features
        self.pool = nn.AdaptiveMaxPool2d(output_size=1)

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    @property
    def out_dim(self) -> int:
        return self._FEATURE_DIM

    def forward(self, tiles: Tensor) -> Tensor:
        features = self.features(tiles)  # (N, 512, H', W')
        pooled = self.pool(features)  # (N, 512, 1, 1)
        return torch.flatten(pooled, start_dim=1)  # (N, 512)


__all__ = ["VGG16Encoder"]
