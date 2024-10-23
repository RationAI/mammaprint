from pathlib import Path

import mlflow
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import data as geometric_data


class GraphDataset(geometric_data.Dataset):
    """Dataset of WSI graphs."""

    def __init__(
        self,
        graphs: list,
        splits: dict[str, float],
        seed: int,
        mode="node",
        *args,
        **kwargs,
    ):
        """Initializes a dataset of WSI graphs.

        Args:
            graphs (list): list of WSI graphs
            splits (dict[str, float]): dictionary of desired splits.
            seed (int): seed
            mode (str, optional): select what label is used for a graph.
                'node' gives graphs with node-level labels.
                'graph' gives graph label.
                Defaults to "node".
            *args: Arguments for torch_geometric.data.Dataset.
            **kwargs : Keyword arguments for torch_geometric.data.Dataset.
        """
        super().__init__(*args, **kwargs)
        self.data = graphs
        self.seed = seed

        split_sum = sum(splits.values())
        if split_sum != 1:
            raise ValueError("Sum of split sizes is not 1!")

        self.splits = splits
        self.mode = mode
        self.preprocess()

    def split(self):
        datasets = {}
        data = self.data
        size_unsplit = 1
        for split_name, size in self.splits.items():
            scaled_size = size / size_unsplit
            size_unsplit -= size
            if scaled_size >= 1:
                datasets[split_name] = GraphDataset(
                    data, splits={split_name: 1}, seed=self.seed
                )
                break
            labels = [graph.graph_label for graph in data]
            split, data = train_test_split(
                data, train_size=scaled_size, stratify=labels, random_state=self.seed
            )
            datasets[split_name] = GraphDataset(
                split, splits={split_name: 1}, seed=self.seed
            )

        return datasets

    def get(self, idx):
        """Get graph on a given index.

        torch_geometric data uses get() instead of __getitem__(). Same for len().
        """
        return self.data[idx]

    def len(self):
        return len(self.data)

    def _standardize(self, x):
        # standardize features
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        return (x - means) / stds

    def preprocess(self):
        # standardize
        for graph in self.data:
            graph.x = self._standardize(graph.x)
            if self.mode == "node":
                graph.y = graph.node_labels
            elif self.mode == "graph":
                graph.y = graph.graph_label


class MlflowGraphDataset(GraphDataset):
    """Graph dataset to use with graphs saved in MLFlow."""

    def __init__(self, data_uris: list[str], splits, seed, *args, **kwargs):
        data = []
        for uri in data_uris:
            path = mlflow.artifacts.download_artifacts(uri)
            path = Path(path)
            data.extend(torch.load(file) for file in sorted(path.iterdir()))

        super().__init__(data, splits, seed, *args, **kwargs)
