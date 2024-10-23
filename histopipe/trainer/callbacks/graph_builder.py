# Standard Imports
from pathlib import Path
from typing import Any

# Third-Party Imports
import lightning
import mlflow
import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric import data
from torch_geometric.typing import OptTensor
from torch_geometric.utils import remove_isolated_nodes

# Local Imports
from histopipe.trainer.callbacks.dataloader_agnostic import DataloaderAgnosticCallback


class SlideGraph(data.Data):
    """an object representing a slide as a graph.

    A slide is cut into tiles without overlap.
    These tiles are first used as an input for feature extractor
    Extracted features are then used as nodes of a graph.

    Attributes:
        x: a torch Tensor of graph nodes of shape [n_nodes, n_features]
        edge_index: Tensor of shape [2, n_edges], representing all edges in a graph
        edge_attr: Optinal tensor of edge features.
        node_labels: a target label for each node. Shape is [n_nodes, n_classes]
        graph_labels: a target label for whole slide.
        metadata: metadata of the original slide
    """

    def __init__(
        self,
        x: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        edge_attr: OptTensor = None,
        node_labels: OptTensor = None,
        graph_label: OptTensor = None,
        metadata: dict[str, any] | None = None,
        **kwargs,
    ):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        self.metadata = metadata
        self.old_indices = None
        self.node_labels = node_labels
        self.graph_label = graph_label

    def remove_blank_nodes(self):
        """Remove background patches from graph_representation.

        By default, the WSI graph exactly corresponds to the WSI.
        Meaning there are nodes for background patches.
        This method removes them, leaving a representation consistent with patches from tiler.
        The mapping of nodes to the original WSI is kept in 'old_indices'.
        """
        # empty nodes already removed
        if self.old_indices:
            return
        edge_index, edge_attr, mask = remove_isolated_nodes(
            self.edge_index, num_nodes=self.num_nodes
        )
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x = self.x[mask]
        if self.node_labels is not None:
            self.node_labels = self.node_labels[mask]
        self.old_indices = torch.where(mask)[0]


class GraphBuilder(DataloaderAgnosticCallback):
    """Callback used to construct a SlideGraph during testing stage.

    To use this callback, start a test run with a trained feature extractor on non-overlaping tiled dataset.
    Resulting graphs are uploaded to mlflow run as artifacts.
    """

    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.slide_metadata = None

    @staticmethod
    def _preprocess_data(data: torch.Tensor) -> list[NDArray]:
        return data.detach().cpu()

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        self.row = []
        self.col = []
        self.outputs = []
        self.y = []
        self.slide_metadata = metadata
        # use this to map extracted features to nodes
        # eg. if patch coords are (512, 1024), map to (1, 2)
        self.scale_factor = (
            (metadata["tile_size"] * 2 ** metadata["sample_level"]).cpu().numpy()
        )
        self.slide_metadata["scale_factor"] = self.scale_factor

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save extracted features with corresponding x/y coordinates."""
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        _, y, metadata = batch
        row = self._preprocess_data(metadata["coord_y"]).to(int) // self.scale_factor
        col = self._preprocess_data(metadata["coord_x"]).to(int) // self.scale_factor
        outputs = self._preprocess_data(outputs["outputs"])
        y = self._preprocess_data(y)

        self.row.append(row)
        self.col.append(col)
        self.y.append(y)
        self.outputs.append(outputs)

    @staticmethod
    def _get_shift(array, shift_y, shift_x):
        """Shift 2d bool array.

        Resulting array is used to get patch neighborhood edges in a given direction

        Args:
            array (torch.Tensor[bool]): map of non-zero tiles (result of filtering during tiling)
            shift_y (int): shift along y-axis - used to get up-down edges
            shift_x (int): shift along x-axis - used to get right-left edges

        Returns:
            np.array: array shifted
        """
        shifted = np.roll(array, (shift_y, shift_x), (0, 1))
        if shift_y < 0:
            shifted[shift_y:, :] = False
        else:
            shifted[:shift_y, :] = False
        if shift_x < 0:
            shifted[:, shift_x:] = False
        else:
            shifted[:, :shift_x] = False
        return shifted

    @staticmethod
    def _get_direction_edges(array, shift_y, shift_x):
        """Get edges in a given direction.

        shift_y, shift_x == 0, 1 -> left - right edges
        shift_y, shift_x == 1, 0 -> up - down edges
        shift_y, shift_x == 1, -1 -> diagonal up - down/right - left edges

        Args:
            array (torch.Tensor[bool]): bool map of non-zero tiles
            shift_y (int): shift along y-axis - used to get up-down edges
            shift_x (int): shift along x-axis - used to get right-left edges

        Returns:
            tuple[np.array[int], np.array[int]]: indices of nodes that share an edge
        """
        shifted = GraphBuilder._get_shift(array, shift_y, shift_x)
        edge_to_y, edge_to_x = np.where(array & shifted)
        edge_from_x = edge_to_x - shift_x
        edge_from_y = edge_to_y - shift_y
        edge_from = np.ravel_multi_index((edge_from_y, edge_from_x), array.shape)
        edge_to = np.ravel_multi_index((edge_to_y, edge_to_x), array.shape)
        return edge_from, edge_to

    @staticmethod
    def get_edge_index(array):
        """Get all edges of a graph derived from array.

        Returns:
            2 tensors of flattened node indices.

        Example:
        input array (1 == True, 0 == False)
        [
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
        ]
        returns edges
        [ 4,  4,  4,  5,  5,  5,  7,  7,  7,  7, 10, 10]
        [ 4,  5,  7,  4,  5,  7,  4,  5,  7, 10,  7, 10]

        Args:
            array (torch.Tensor[bool]): bool map of non-zero tiles

        Returns:
            tuple[torch.Tensor[int], torch.Tensor[int]]: edge index
        """
        all_edges_from = []
        all_edges_to = []
        for y in [-1, 0, 1]:
            for x in [-1, 0, 1]:
                dir_edges_from, dir_edges_to = GraphBuilder._get_direction_edges(
                    array, y, x
                )
                all_edges_from.append(dir_edges_from)
                all_edges_to.append(dir_edges_to)
        all_edges_from, all_edges_to = (
            np.concatenate(all_edges_from),
            np.concatenate(all_edges_to),
        )
        sort = np.lexsort((all_edges_to, all_edges_from))
        all_edges_from = torch.from_numpy(all_edges_from[sort])
        all_edges_to = torch.from_numpy(all_edges_to[sort])
        return torch.stack((all_edges_from, all_edges_to)).to(int)

    def get_nodes(self):
        """Return extracted features mapped to tensor.

        Returns:
            torch.Tensor: Tensor representation of a slide.
        """
        max_row = self.row.max() + 1
        max_col = self.col.max() + 1
        output_dim = self.outputs[0].numel()
        nodes = torch.zeros((max_row, max_col, output_dim))
        nodes[self.row, self.col] = self.outputs
        nodes = torch.Tensor(nodes)
        return nodes

    def get_node_labels(self):
        """Return patch labels mapped to graph nodes.

        Returns:
           torch.Tensor : Patch labels mapped to graph
        """
        max_row = self.row.max() + 1
        max_col = self.col.max() + 1
        y = torch.cat(self.y)
        label_dim = y.shape[-1]
        label_map = np.zeros((max_row, max_col, label_dim))
        label_map[self.row, self.col] = y
        return torch.Tensor(label_map).reshape(-1, label_dim)

    def _cat_data(self):
        """After test dataloader ends, collected data is in form of list[tensor]. we need tensors."""
        self.outputs = torch.cat(self.outputs)
        self.row = torch.cat(self.row)
        self.col = torch.cat(self.col)

    def _get_graph(self):
        """Takes saved predictions and uses them to create a graph of WSI.

        Returns:
            SlideGraph: final graph representation of WSI.
        """
        self._cat_data()
        nodes = self.get_nodes()

        self.slide_metadata["node_grid_shape"] = tuple(nodes.shape)

        patch_map = torch.any(nodes, dim=2)  # bool map of patches
        edge_index = GraphBuilder.get_edge_index(patch_map)

        nodes = nodes.flatten(0, -2)

        node_labels = self.get_node_labels().to(int)
        graph_label = self.slide_metadata["class_id"].to(int)

        slidegraph = SlideGraph(
            x=nodes,
            edge_index=edge_index,
            node_labels=node_labels,
            graph_label=graph_label,
            metadata=self.slide_metadata,
        ).to("cpu")
        slidegraph.remove_blank_nodes()
        return slidegraph

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        """Features of all patches extracted, time to assemble the graph."""
        # this graph is a graph representation of the entire wsi
        # meaning the white background patches are present as zero tensors
        # they are removed in the next step with remove_blank_nodes.
        # you can add them back later,

        slidegraph = self._get_graph()

        slide_name = self.slide_metadata["slide_name"]
        path = self.save_dir / slide_name
        torch.save(slidegraph, f=path)
        mlflow.log_artifact(path, "graphs")
