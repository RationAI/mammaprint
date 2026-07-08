"""Per-tile encoders: identity (embeddings path) and image backbones."""

from ml.models.encoders.base import Encoder
from ml.models.encoders.identity import IdentityEncoder
# from ml.models.encoders.vgg16 import VGG16Encoder
# from ml.models.encoders.virchow2 import Virchow2Encoder


__all__ = [
            "Encoder", 
           "IdentityEncoder", 
         #   "VGG16Encoder", 
         #   "Virchow2Encoder"
           ]
