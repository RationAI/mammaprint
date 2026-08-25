import torch

from ml.models.aggregators import MeanPool
from ml.models.encoders import IdentityEncoder
from ml.models.heads import LinearHead
from ml.models.module import MammaprintModule
from ml.typing import PyramidBag, PyramidSample, PyramidSlideMetadata


encoder: IdentityEncoder[PyramidBag] = IdentityEncoder(out_dim=16)
module: MammaprintModule[PyramidBag, PyramidBag] = MammaprintModule(
    encoder=encoder,
    aggregator=MeanPool(feature_dim=16),
    head=LinearHead(in_dim=16),
)
metadata: PyramidSlideMetadata = {
    "slide_id": "typed-pyramid",
    "anchors": ((0, 0),),
}
sample: PyramidSample = (
    PyramidBag(torch.randn(1, 21, 16)),
    torch.tensor([1.0]),
    metadata,
)
module.predict_step([sample], batch_idx=0)
