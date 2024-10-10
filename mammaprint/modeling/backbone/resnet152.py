import torchvision
from torch import nn


def resnet152(weights: str | None = None) -> nn.Module:
    resnet = torchvision.models.resnet152(weights=weights)
    return nn.Sequential(*(list(resnet.children())[:-2]))
