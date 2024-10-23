from torch_geometric import data as geom_data


class GraphSamplerInitializer:
    """Some samplers from torch_geometric need to be initialized with the dataset.

    See ImbalancedSampler as an example.
    """

    def __new__(cls, dataset, sampling_type, sampler_partial):
        if sampling_type == "dataset":
            # could be used for sampling graphs in graph classification
            return dataset, sampler_partial(dataset)
        if sampling_type == "node":
            # all graphs in dataset merged into one bit disconnected graph
            # good for balanced node sampling
            merged_graphs = geom_data.Batch.from_data_list(dataset)
            return merged_graphs, sampler_partial(merged_graphs)
