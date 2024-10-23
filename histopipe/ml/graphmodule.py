import torch

from histopipe.ml.histopipemodule import HistoPipeModule


class GraphModule(HistoPipeModule):
    """HistoPipeModule subclass to allow work with graphs.

    Graph dataset batching works on torch_geometric.data.Data objects.
    These objects have x, edge_index, edge_attrs, etc. attributes.
    We need to allow torch_geometric to do it's graph batching and then unpack for model input.
    """

    def forward(self, x):
        x = self.model(*x)
        x = self.output_activation(x)
        return x

    def _unpack_batch_to_inputs(self, batch):
        x = batch.x, batch.edge_index
        y = batch.y.to(torch.float32)
        metadata = batch.metadata
        return x, y, metadata

    def training_step(self, batch, batch_idx):
        batch = self._unpack_batch_to_inputs(batch)
        super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        batch = self._unpack_batch_to_inputs(batch)
        super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        batch = self._unpack_batch_to_inputs(batch)
        super().test_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch = self._unpack_batch_to_inputs(batch)
        super().predict_step(batch, batch_idx, dataloader_idx)
