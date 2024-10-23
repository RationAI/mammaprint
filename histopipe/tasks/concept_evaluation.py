import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars
import sklearn
import sklearn.preprocessing
import torch
from scipy import stats
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm

from histopipe.tasks.task import AbstractTask


log = logging.getLogger("TCAV")


class ConceptEvaluation(AbstractTask):
    """Class for evaluating concepts completeness and TCAV scores.

    Args:
        ml (object): ML model.
        datamodule (object): Data module.
        hyperparameters (dict): Hyperparameters.
        experiment_name (str): Name of the experiment.
        run_name (str): Name of the run.
        expl_model_activations_mlflow_uri (str): URI for explanatory model activations.
        concept_discovery_activations_mlflow_uri (str): URI for concept discovery activations.
        classif_labels_mlflow_uri (str): URI for classification labels.
        metadata_mlflow_uri (str): URI for metadata.
        number_of_random_sets (int): Number of random sets.
        activation_col (str): Activation column name.
        concept_label_col (str): Concept label column name.
        class_label_col (str): Class label column name.
        join_columns (str): Columns used for joining.
        concept_vectors_mlflow_uri (str): URI for concept vectors.
        output_index (int): Output index.
        stage (str): Model stage.
        use_k_means_centroids (bool): Flag for using K-means centroids.
        layer_idx (int): Layer index.
        compute_tcav (bool): Flag for computing TCAV.
        compute_completness (bool): Flag for computing completeness.
    """

    def __init__(
        self,
        ml,
        datamodule,
        hyperparameters,
        experiment_name,
        run_name,
        expl_model_activations_mlflow_uri,
        concept_discovery_activations_mlflow_uri,
        classif_labels_mlflow_uri,
        metadata_mlflow_uri,
        number_of_random_sets,
        activation_col,
        concept_label_col,
        class_label_col,
        join_columns,
        concept_vectors_mlflow_uri,
        output_index,
        stage,
        layer_idx,
        compute_tcav=True,
        use_k_means_centroids=False,
        compute_completness=True,
    ):
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        self.ml = ml
        self.output_index = output_index
        self.compute_tcav = compute_tcav
        self.compute_completness = compute_completness
        self.datamodule = datamodule
        self.hyperparameters = hyperparameters
        self.expl_model_activations_mlflow_uri = expl_model_activations_mlflow_uri
        self.concept_discovery_activations_mlflow_uri = (
            concept_discovery_activations_mlflow_uri
        )
        self.metadata_mlflow_uri = metadata_mlflow_uri
        self.number_of_random_sets = number_of_random_sets
        self.stage = stage
        self.classif_labels_mlflow_uri = classif_labels_mlflow_uri
        self.activation_col = activation_col
        self.concept_label_col = concept_label_col
        self.class_label_col = class_label_col
        self.join_columns = join_columns
        self.concept_vectors_mlflow_uri = concept_vectors_mlflow_uri
        self.layer_idx = layer_idx
        self.use_k_means_centroids = use_k_means_centroids

        if self.classif_labels_mlflow_uri is None and self.compute_completness:
            raise ValueError(
                "For completness calculation you have to provide class labels. (classif_labels_mlflow_uri)."
            )

    def setup(self):
        super().setup()

    def run(self):
        self.__load_acts()
        if self.compute_tcav:
            if self.use_k_means_centroids:
                cavs = self.__train_cavs_as_centroids(random=False)
                random_cavs = self.__train_cavs_as_centroids(random=True)
            else:
                cavs = self.__train_cavs(random=False)
                random_cavs = self.__train_cavs(random=True)

            cavs.update(random_cavs)
            log.info(f"Found following concept ids: {list(cavs.keys())}")
            _ = self.__compute_tcav_scores(cavs)

        if self.compute_completness:
            if self.concept_vectors_mlflow_uri is not None:
                concept_vectors = np.load(
                    mlflow.artifacts.download_artifacts(
                        artifact_uri=self.concept_vectors_mlflow_uri
                    )
                )
            else:
                concept_vectors = np.vstack(
                    [np.vstack(cavs[k]).mean(axis=0) for k in cavs]
                )
                concept_vectors = concept_vectors[:-1, :]  # exclude random

            self.__evaluate_completness(concept_vectors)
            self.__evaluate_completness(
                concept_vectors,
                scores=False,
                descr="Classif on hard concept assignments",
            )
        self.__plot_result()

    def teardown(self):
        super().teardown()

    def __load_acts(self):
        """Load precomputed activations from mlflow run."""
        log.info("Loading activations")
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=self.concept_discovery_activations_mlflow_uri
        )
        concepts_acts = polars.read_parquet(path)
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=self.expl_model_activations_mlflow_uri
        )
        self.clasif_acts = polars.read_parquet(path)
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=self.metadata_mlflow_uri
        )
        self.metadata = polars.read_parquet(path)

        concepts_acts = concepts_acts.drop(self.activation_col)
        self.concepts_acts = concepts_acts.join(
            self.clasif_acts, on=self.join_columns, how="inner"
        )
        log.info(f"Loaded activations, shape {self.concepts_acts.shape}")
        if len(self.concepts_acts) != len(self.clasif_acts) or len(
            self.concepts_acts
        ) != len(concepts_acts):
            raise ValueError(
                "Number of activations from classif and concepts dicovery must be the same as they must be from same dataset"
            )
        if self.classif_labels_mlflow_uri is not None:
            path = mlflow.artifacts.download_artifacts(
                artifact_uri=self.classif_labels_mlflow_uri
            )
            class_labels = polars.read_parquet(path)
            len_orig = len(self.concepts_acts)
            self.concepts_acts = self.concepts_acts.join(
                class_labels, on=self.join_columns, how="inner"
            )
            if (
                len(self.concepts_acts) != len(class_labels)
                or len(self.concepts_acts) != len_orig
            ):
                raise ValueError(
                    "Number of activations from classif and concepts dicovery must be the same as they must be from same dataset"
                )

    def __train_cavs_as_centroids(self, random=False):
        """Obtain concept vectors using alternative centroid method.

        Args:
            random (bool): If true, random concept vector is created by sampling random concept examples
        Returns:
            cavs (dict[str, list[NdArray]]) dictionary containing concept vectors for concepts.
        """
        unique_labels = (
            self.concepts_acts.select(self.concept_label_col)
            .unique()[self.concept_label_col]
            .sort()
        )
        cavs = {}

        log.info("Trainig concept classifiers")
        for label in tqdm(unique_labels if not random else ["random"]):
            if random:
                non_concept_acts = np.vstack(
                    self.concepts_acts.sample(
                        fraction=1 / len(unique_labels), shuffle=True
                    )[self.activation_col]
                )
                curr_cavs = [non_concept_acts.mean(axis=0)]
            else:
                curr_concept_acts = self.concepts_acts.filter(
                    self.concepts_acts[self.concept_label_col] == label
                )
                curr_concept_acts = curr_concept_acts.sample(fraction=0.5)
                curr_concept_acts = np.vstack(
                    curr_concept_acts[self.activation_col].to_numpy()
                )
                curr_cavs = [curr_concept_acts.mean(axis=0)]
            cavs[label] = curr_cavs
        return cavs

    def __train_cavs(self, random=False):
        """Obtain concept vectors based on examples using classifier method.

        Args:
            random (bool): If true, random concept vector is created by sampling random concept examples
        Returns:
            cavs (dict[str, list[NdArray]]) dictionary containing concept vectors for concepts.
        """
        # Get unique labels
        unique_labels = np.sort(
            self.concepts_acts.select(self.concept_label_col).unique()[
                self.concept_label_col
            ]
        )
        cavs = {}
        log.info("Trainig concept classifiers")
        for label in tqdm(unique_labels if not random else ["random"]):
            curr_cavs = []
            if random:
                # random vs random concept
                curr_concept_acts = self.concepts_acts.sample(
                    fraction=1 / len(unique_labels), shuffle=True
                )
                curr_concept_acts = np.vstack(
                    curr_concept_acts[self.activation_col].to_numpy()
                )
                non_concept_acts = self.concepts_acts.sample(
                    fraction=1 / len(unique_labels), shuffle=True
                )
            else:
                curr_concept_acts = self.concepts_acts.filter(
                    self.concepts_acts[self.concept_label_col] == label
                )
                curr_concept_acts = np.vstack(
                    curr_concept_acts[self.activation_col].to_numpy()
                )
                non_concept_acts = self.concepts_acts.filter(
                    self.concepts_acts[self.concept_label_col] != label
                )

            for _ in tqdm(range(self.number_of_random_sets)):
                if random:
                    # resample random concept every time
                    curr_concept_acts = self.concepts_acts.sample(
                        fraction=1 / len(unique_labels), shuffle=True
                    )
                    curr_concept_acts = np.vstack(
                        curr_concept_acts[self.activation_col].to_numpy()
                    )
                random_acts = non_concept_acts.sample(n=curr_concept_acts.shape[0])
                # random_acts = non_concept_acts
                random_acts = np.vstack(random_acts[self.activation_col].to_numpy())
                train_x = np.vstack([curr_concept_acts, random_acts])
                train_y = np.hstack(
                    [
                        np.ones(curr_concept_acts.shape[0]),
                        np.zeros(random_acts.shape[0]),
                    ]
                )

                # shuffle
                shuffled_indices = np.random.permutation(np.arange(train_x.shape[0]))
                train_x = train_x[shuffled_indices]
                train_y = train_y[shuffled_indices]
                clf, _ = self.__train_classif(
                    train_x, train_y, descr=f"Concept {label} vs random"
                )
                curr_cavs.append(clf.coef_.flatten())
            cavs[label] = curr_cavs
        return cavs

    def __compute_tcav_scores(self, cavs):
        log.info("Computing TCAV score")
        global gradient
        gradient = None
        score_acum = np.zeros((len(cavs.keys()), self.number_of_random_sets))
        dot_prod_acum = np.zeros((len(cavs.keys()), self.number_of_random_sets))

        def grad_hook(module, grad_input, grad_output):
            global gradient
            gradient = grad_output

        list(self.ml.model.children())[self.layer_idx].register_full_backward_hook(
            grad_hook
        )
        self.datamodule.setup()
        self.datamodule.datasets["test"].generate_samples()
        self.ml.model.eval()
        act = self.ml.output_activation

        for concept_idx, concept in enumerate(cavs.keys()):
            pred = (
                act(
                    self.ml.model.classifier(
                        torch.tensor(np.vstack(cavs[concept]).mean(axis=0))
                        .unsqueeze(dim=0)
                        .cuda()
                    )
                )
                .cpu()
                .detach()
                .numpy()
                .flatten()[self.output_index]
            )
            mlflow.log_metric("classif prediction on concept vector", pred)
            print(f"Cocnept {concept} (index {concept_idx}) prediction {pred}")

        # compute TCAV score
        count = 0
        for sample in tqdm(
            self.datamodule.datasets["test"],
            total=len(self.datamodule.datasets["test"]),
        ):
            input, label, meta = sample
            count += 1
            pred = self.ml.model(input.cuda())[self.output_index]
            pred = act(pred)
            pred.backward()
            for concept_idx, concept in enumerate(cavs.keys()):
                dot_prod = np.dot(np.vstack(cavs[concept]), gradient[0].cpu().numpy())
                mlflow.log_metric(
                    f"concept {concept} dot product with grad", dot_prod.mean(), step=1
                )
                score_acum[concept_idx] += dot_prod > 0
                dot_prod_acum[concept_idx] += dot_prod
        tcavs = score_acum / count
        grad_dot_prod = dot_prod_acum / count

        # get p values and log to mlflow
        print(tcavs)
        for concept_idx, concept in enumerate(cavs.keys()):
            t_statistic, p_value = stats.ttest_ind(
                tcavs[concept_idx], tcavs[list(cavs.keys()).index("random")]
            )
            mlflow.log_metric(
                f"TCAV score concept {concept}", tcavs[concept_idx].mean()
            )
            mlflow.log_metric(
                f"TCAV score concept std {concept}", tcavs[concept_idx].std()
            )
            mlflow.log_metric(f"pval concept {concept} vs random", p_value)

            # test if TCAV can be equal to 0.5
            t_statistic, p_value = stats.ttest_1samp(tcavs[concept_idx], 0.5)
            mlflow.log_metric(f"pval concept {concept} TCAV equals 0.5 ", p_value)
            mlflow.log_metric(
                f"Mean dot product with gradient concept {concept}",
                grad_dot_prod[concept_idx].mean(),
            )
            log.info(
                f"concept: {concept}, mean TCAV score : {tcavs[concept_idx].mean()}, p value {p_value} mean dot prod with grad {grad_dot_prod[concept_idx].mean()}"
            )
        return tcavs

    def __train_classif(self, all_x, all_y, descr="", clf=None):
        """Train a classifier using stochastic gradient descent (SGD) and evaluate its performance.

        Args:
            all_x (numpy.ndarray): Input features for training and testing.
            all_y (numpy.ndarray): Labels corresponding to the input features.
            descr (str, optional): Description for the training and evaluation. Default is an empty string.
            clf (sklearn.base.BaseEstimator, optional): Classifier object. If None, a new SGDClassifier will be created.

        Returns:
            clf (sklearn.base.BaseEstimator): Trained classifier.
            f_score (float): F1 score of the classifier on the test data.

        """
        # split
        split_idx = int(all_x.shape[0] * 0.8)
        test_x = all_x[split_idx:]
        test_y = all_y[split_idx:]
        train_x = all_x[:split_idx]
        train_y = all_y[:split_idx]

        if clf is None:
            # train
            clf = SGDClassifier(class_weight="balanced", max_iter=100)
            clf.fit(train_x, train_y)

        # eval
        y_test_pred = clf.predict(test_x)
        report = classification_report(test_y, y_test_pred)
        log.info(descr + "\n" + str(report))
        f_score = f1_score(test_y, y_test_pred)
        mlflow.log_metric(f"{descr} f1 score", f_score, step=1)
        return clf, f_score

    def __evaluate_completness(
        self, concept_vectors, scores=True, descr="Classif on concept scores"
    ):
        """Evaluate the completeness of a set of concepts in context of explaining the classifier.

        Args:
            concept_vectors (numpy.ndarray): The vectors representing concepts.
            scores (bool, optional): If True, use concept scores for evaluation. If False, use labels.
            descr (str, optional): Description for the metrics.

        Returns:
                None : Results are logged into mlflow
        """
        y = np.vstack(self.concepts_acts[self.class_label_col]).flatten()
        act_x = np.vstack(self.concepts_acts[self.activation_col])
        if scores:
            concept_x = np.dot(act_x, concept_vectors.T) / (
                np.linalg.norm(act_x, axis=1)[:, None]
                * np.linalg.norm(concept_vectors, axis=1)
            )
        else:
            concept_x = np.vstack(self.concepts_acts["label"]).flatten().astype(int)
            label_binarizer = sklearn.preprocessing.LabelBinarizer()
            label_binarizer.fit(range(max(concept_x) + 1))
            concept_x = label_binarizer.transform(concept_x)

        shuffled_indices = np.random.permutation(np.arange(act_x.shape[0]))
        act_x = act_x[shuffled_indices]
        concept_x = concept_x[shuffled_indices]
        y = y[shuffled_indices]
        log.info(f"Training classifier on activations on {len(act_x)} examples")
        clf, f_score_full = self.__train_classif(act_x, y, "Classif on activations ")

        # Perform cross-validation and print the accuracy for each fold
        kf = KFold(n_splits=20, shuffle=True, random_state=42)
        cv_results = cross_val_score(clf, act_x, y, cv=kf)
        print("Cross-validation results:", cv_results)
        print(f"Mean accuracy: {cv_results.mean():.4f}")
        print(f"Stdev. accuracy: {cv_results.std():.4f}")

        log.info(f"Training classifier on concepts {len(concept_x)} examples")
        clf, f_score_concept = self.__train_classif(concept_x, y, descr)
        completnes = (f_score_concept - 1 / len(np.unique(y))) / (
            f_score_full - 1 / len(np.unique(y))
        )
        mlflow.log_metric("completnes score", completnes)
        log.info(f"Weight vector {clf.coef_}")

        # Perform cross-validation and print the accuracy for each fold
        kf = KFold(n_splits=20, shuffle=True, random_state=42)
        cv_results = cross_val_score(clf, concept_x, y, cv=kf)
        print("Cross-validation results:", cv_results)
        print(f"Mean accuracy: {cv_results.mean():.4f}")
        print(f"Stdev. accuracy: {cv_results.std():.4f}")

    def __plot_result(self):
        """Plots overlaps of concepts with annotations and class prportion per concepts."""
        self.concepts_acts = self.concepts_acts.with_columns(
            polars.col("label").cast(polars.Int64, strict=False)
        )
        res = self.concepts_acts.groupby(["label"]).mean()
        plt.bar(res["label"], res["class_id"] * 100)
        plt.title("Proportion of positive examples for each concepts")
        plt.xlabel("Concept id")
        plt.ylabel("% of positive examples")
        plt.savefig("posit_examples_per_concepts.png")
        mlflow.log_artifact("posit_examples_per_concepts.png")

        plt.clf()
        plt.bar(res["label"], res["annot_coverage"] * 100)
        plt.title("Cancer annotation coverage")
        plt.xlabel("Concept id")
        plt.ylabel("% of cancer annotation coverage")
        plt.savefig("anot_covrg_concepts.png")
        mlflow.log_artifact("anot_covrg_concepts.png")
