import numpy as np

from histopipe.trainer.callbacks.heatmap_visualizer import HeatmapVisualizer


class GraphHeatmapVisualizer(HeatmapVisualizer):
    """Callback to visualize graph prediction using HeatmapVisualizer.

    The only thing this does is convert graph node predictions to a format expected by HeatmapVisualzer.
    """

    def _get_coords(self, graph, metadata):
        scale_factor = metadata["scale_factor"]
        h, w, _ = metadata["node_grid_shape"]
        y, x = np.unravel_index(graph.old_indices.cpu(), (h, w)) * scale_factor
        return x, y

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        graph = batch
        unbatched_metadata = [
            {k: v[i] for k, v in batch.metadata.items()}
            for i in range(batch.num_graphs)
        ]
        metadata = unbatched_metadata[0]
        metadata["coord_x"], metadata["coord_y"] = self._get_coords(graph, metadata)
        metadata["slide_channels"] = outputs["outputs"].shape[1]
        if dataloader_idx != self._current_dataloader_idx:
            self.on_test_dataloader_start(
                trainer=trainer,
                pl_module=pl_module,
                metadata=metadata,
                dataloader_idx=dataloader_idx,
            )

            self._current_dataloader_idx = dataloader_idx
        return super().on_test_batch_end(
            trainer,
            pl_module,
            outputs,
            (None, None, metadata),
            batch_idx,
            dataloader_idx,
        )
