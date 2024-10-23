import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from histopipe.tasks.task import AbstractTask


log = logging.getLogger("concept_discovery")


class ConceptDiscoveryTask(AbstractTask):
    """A class for performing concept discovery using KMeans clustering on activations.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        run_name (str): The name of the MLflow run.
        activations_mlflow_uris (str): MLflow URI for activations data.
        metadata_mlflow_uris (list): List of MLflow URIs for metadata data.
        stage (str): The stage of the concept discovery process.
        image_builder (optional): An image builder for visualization.
        max_cluster (int, optional): The maximum number of clusters to consider.
        cluster_cnt (int, optional): The fixed number of clusters if provided.
        activations_column (str, optional): The column containing model activations.
        hyperparameters (dict, optional): Additional hyperparameters for the task.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name,
        activations_mlflow_uris: str,
        metadata_mlflow_uris: list[str],
        stage: str,
        image_builder=None,
        max_cluster: int = 10,
        cluster_cnt=None,
        activations_column="model_output",
        hyperparameters: dict | None = None,
    ):
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        self.image_builder = image_builder
        self.activations_column = activations_column
        self.metadata_mlflow_uris = metadata_mlflow_uris
        self.max_cluster = max_cluster
        self.activations_mlflow_uris = activations_mlflow_uris
        self.cluster_cnt = cluster_cnt

    def setup(self):
        super().setup()

    def run(self):
        tiles_data, metadata = self.__download_activations()
        acts = np.vstack(tiles_data[self.activations_column].to_numpy())

        if self.cluster_cnt is None:
            self.cluster_cnt = self.__find_cluster_cnt(acts)
        tiles_data = self.__asign_and_save_clusters(tiles_data, acts)

        if self.image_builder is not None:
            log.info("Visualizing clusters on slides")
            self.__visualize_clusters(tiles_data, metadata)

    def teardown(self):
        super().teardown()

    def __download_activations(self):
        """Donwnload activations from mlflow."""
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=self.activations_mlflow_uris[0]
        )
        tiles_data = polars.read_parquet(path)
        for tiles_uri in self.activations_mlflow_uris[1:]:
            path = mlflow.artifacts.download_artifacts(artifact_uri=tiles_uri)
            df = polars.read_parquet(path)
            tiles_data = polars.concat([tiles_data, df])

        path = mlflow.artifacts.download_artifacts(
            artifact_uri=self.metadata_mlflow_uris[0]
        )
        metadata = polars.read_parquet(path)
        for meta_uri in self.metadata_mlflow_uris[1:]:
            path = mlflow.artifacts.download_artifacts(artifact_uri=meta_uri)
            df = polars.read_parquet(path)
            metadata = polars.concat([metadata, df])
        metadata.write_parquet("slides.parquet")
        mlflow.log_artifact(local_path="slides.parquet")

        return tiles_data, metadata

    def __find_cluster_cnt(self, acts):
        """Determine the optimal number of clusters for KMeans clustering using the elbow method.

        Args:
            acts (array-like): Input data for clustering.

        Returns:
        int: The optimal number of clusters determined using the elbow method.
        """
        inertia = []
        cluster_range = range(2, self.max_cluster)
        for n_clusters in tqdm(cluster_range):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            kmeans.fit(acts)
            inertia.append(kmeans.inertia_)
        kn = KneeLocator(
            x=list(cluster_range),
            y=inertia,
            S=1.0,
            curve="convex",
            direction="decreasing",
        )
        log.info(f"Number of cluster {kn.knee}")
        return kn.knee

    def __asign_and_save_clusters(self, tiles_data, acts):
        kmeans = KMeans(n_clusters=self.cluster_cnt, n_init="auto")
        print("Fitting K-Means data.")
        kmeans.fit(acts)
        self.centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        print("Saving concept vectors.")
        tiles_data = tiles_data.with_columns(label=polars.lit(labels))
        np.save("./concept_vectors.npy", self.centroids)
        mlflow.log_artifact(local_path="concept_vectors.npy")
        tiles_data.write_parquet("tiles_with_labels.parquet")
        mlflow.log_artifact(local_path="tiles_with_labels.parquet")
        return tiles_data

    def __visualize_clusters(self, tiles_data, metadata):
        """Visualize clusters overlays.

        Parameters:
            tiles_data (pandas.DataFrame): DataFrame containing tile data, including 'slide_name', 'label', etc.
            metadata (pandas.DataFrame): DataFrame containing metadata for slides.

        Returns:
        None : overlays are saved to mlflow
        """
        # join tiles with meta data
        print("Performing join operation on data.")
        tiles_data = tiles_data.join(metadata, on="slide_name", how="left")
        cmap = plt.cm.Set1
        norm = plt.Normalize(
            tiles_data["label"].to_numpy().min(), tiles_data["label"].to_numpy().max()
        )

        for _, slide_name in tqdm(
            enumerate(tiles_data["slide_name"].unique().to_list())
        ):
            # Only keep tiles for current slide
            print("Filtering tiles data.")
            current_preds = tiles_data.filter(polars.col("slide_name") == slide_name)
            current_preds = current_preds.sort("label")
            # save each cluster map into a separate bitmap mask
            for cluster_id, cluster_df in enumerate(
                current_preds.partition_by(by="label")
            ):
                print(
                    f"[{slide_name[:10]}] Computing cluster no. {cluster_id} of {self.cluster_cnt}."
                )
                current_meta = cluster_df[0].to_dict(as_series=False)
                current_meta = {k: v[0] for k, v in current_meta.items()}
                current_meta["slide_channels"] = 1
                current_labels = np.ones((len(cluster_df), 1))

                image_builder = self.image_builder(
                    metadata=current_meta, save_dir=f"./cluster_{cluster_id}/"
                )
                image_builder.update(data=current_labels, metadata=cluster_df)
                image_builder.save()
                mlflow.log_artifacts(
                    local_dir=f"./cluster_{cluster_id}",
                    artifact_path=f"cluster_{cluster_id}",
                )

            # Save composite cluster map with RGB values averaged on overlaps
            tiles_data = tiles_data.with_columns(slide_channels=3)

            current_preds.with_columns(slide_channels=3)
            current_meta = current_preds[0].to_dict(as_series=False)
            current_meta = {k: v[0] for k, v in current_meta.items()}
            current_meta["slide_channels"] = 3
            image_builder = self.image_builder(
                metadata=current_meta, save_dir="./composite/"
            )

            current_labels = current_preds["label"].to_numpy()
            rgb_arr = (cmap(norm(current_labels))[:, :3] * 255).astype("uint8")[:, ::-1]
            image_builder.update(data=rgb_arr, metadata=current_preds)
            image_builder.save()
            mlflow.log_artifacts(local_dir="./composite", artifact_path="composite")

        self.__create_similarity_matrix(tiles_data, cmap, norm)

    def __create_similarity_matrix(self, tiles_preds, cmap, norm):
        """Create a similarity matrix visualization based on cluster centroids and label information.

        Args:
            tiles_preds (DataFrame): DataFrame containing prediction information for each tile.
            cmap (matplotlib.colors.Colormap): Colormap for mapping labels to colors.
            norm (matplotlib.colors.Normalize): Normalization for mapping labels to colors.

        Returns:
        None : Cmap is saved to mlflow
        """
        log.info("Creating similarity matrix.")
        unique_labels, unique_indices = np.unique(
            tiles_preds["label"], return_index=True
        )
        rgb_arr = (cmap(norm(tiles_preds["label"]))[:, :3] * 255).astype(np.uint8)[
            :, ::-1
        ]
        unique_rgb_arr = rgb_arr[unique_indices][:, ::-1]
        normalized_rgb_arr = unique_rgb_arr / 255.0

        similarity_matrix = cosine_similarity(self.centroids)
        fig, ax = plt.subplots()
        plt.imshow(similarity_matrix, cmap="viridis")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        for i, color in enumerate(normalized_rgb_arr):
            text = ax.text(
                i,
                -1,
                f"cluster {i}",
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                fontweight="bold",
            )
            text.set_bbox({"facecolor": color, "alpha": 0.9, "edgecolor": color})
            text = ax.text(
                -1.3,
                i,
                f"cluster {i}",
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                fontweight="bold",
            )
            text.set_bbox({"facecolor": color, "alpha": 0.9, "edgecolor": color})
        plt.savefig("similarity_matrix.png")
        mlflow.log_artifact(local_path="similarity_matrix.png")
