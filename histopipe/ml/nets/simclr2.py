import mlflow
import torch


class SimCLR2Extraction(torch.nn.Module):
    def __init__(self, extractor_uri, projection_uri, layer_idx=5):
        super().__init__()
        self.extractor = self._load_external_model(extractor_uri)
        self.maxpool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.extractor.fc = self._load_external_model(projection_uri)
        self.layer_idx = layer_idx

    def forward(self, x):
        x = self.extractor.net(x)
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.extractor.fc(x) if self.training else self._output_cut(x)

    def _output_cut(self, x):
        for layer in self.extractor.fc.layers[: self.layer_idx]:
            x = layer(x)
        return x

    def _load_external_model(self, model_uri):
        return mlflow.pytorch.load_model(model_uri)
