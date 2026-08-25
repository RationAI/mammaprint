from ml.models.aggregators import MeanPool
from ml.models.encoders import IdentityEncoder
from ml.models.heads import LinearHead
from ml.models.module import MammaprintModule
from ml.typing import PyramidBag


encoder: IdentityEncoder[PyramidBag] = IdentityEncoder(out_dim=16)
MammaprintModule(
    encoder=encoder,
    aggregator=MeanPool(feature_dim=16),
    head=LinearHead(in_dim=16),
)
