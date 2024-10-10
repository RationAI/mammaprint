import torchvision
from torch import nn


def resnet101(weights: str | None = None) -> nn.Module:
    resnet = torchvision.models.resnet101(weights=weights)
    return nn.Sequential(*(list(resnet.children())[:-2]))
