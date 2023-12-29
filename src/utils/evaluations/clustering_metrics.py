"""
	File containg all clustering metrics evaluation
"""
from collections.abc import Callable

import numpy as np
# from numba import jit
from sklearn.metrics import (
    adjusted_mutual_info_score,
    confusion_matrix,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from utils import euclidean_distances

def cluster_acc(
    distances: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    mapped_labels = map_ypred_to_ytrue(
        distances=distances,
        y_true=y_true,
        y_pred=y_pred,
    )

    confusion_matrix_result = cluster_confusion_matrix(
        y_true=y_true,
        y_pred=mapped_labels
    )

    return confusion_matrix_result.diagonal().sum() / confusion_matrix_result.sum()


def map_ypred_to_ytrue(
    distances: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
        function responsible for mapping predicted labels to original labels based on similarity
    """
    unique_y_pred_labels = np.unique(y_pred)

    label_map = {}

    for y_pred_label in unique_y_pred_labels:
        y_pred_distance = distances[:, np.where(y_pred == y_pred_label)[0]]
        idxs = np.where(y_pred_distance == 0)[0]

        y_true_labels, y_true_count = np.unique(
            y_true[idxs], return_counts=True
        )

        most_present_label = y_true_labels[np.argmax(y_true_count)]

        label_map[y_pred_label] = most_present_label

    new_labels = np.array(
        [label_map.get(label, -1) for label in y_pred]
    )

    return new_labels


def cluster_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    'confusion matrix sklearn wrapper'
    return confusion_matrix(y_true, y_pred)


def nmi(y_true: np.ndarray, y_pred: np.ndarray, average_method="arithmetic") -> float:
    'NMI sklearn wrapper'
    return float(normalized_mutual_info_score(y_true, y_pred, average_method=average_method))


def ami(y_true: np.ndarray, y_pred: np.ndarray, average_method="arithmetic") -> float:
    'AMI sklearn wrapper'
    return float(adjusted_mutual_info_score(y_true, y_pred, average_method=average_method))


def silhouette(distances: np.ndarray, y_pred: np.ndarray) -> float:
    'silhouette sklearn wrapper'
    return float(silhouette_score(distances, y_pred, metric="precomputed"))


# @jit(nopython=True, cache=True)
def intra_distance(ci: np.ndarray, distances: np.ndarray) -> float:
    cidx = np.where(ci)[0]
    values = distances[cidx][:, cidx]

    return np.max(values)


# @jit(nopython=True, cache=True)
def inter_distance(ck: np.ndarray, cl: np.ndarray, distances: np.ndarray) -> float:
    cl_idx = np.where(cl)[0]
    ck_idx = np.where(ck)[0]

    return np.sum(distances[ck_idx][:, cl_idx]) / (ck_idx.shape[0] * cl_idx.shape[0])


# @jit(nopython=True, cache=True)
def inter_distance2(ck: np.ndarray, cl: np.ndarray, distances: np.ndarray) -> float:
    cl_idx = np.where(cl)[0]
    ck_idx = np.where(ck)[0]

    return np.min(distances[ck_idx][:, cl_idx])


# @jit(nopython=True, cache=True)
def dunn_helper(
    distances: np.ndarray,
    labels: np.ndarray,
    delta_fast: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        float] = inter_distance
) -> float:
    unique_labels = np.unique(labels)
    total_labels = unique_labels.shape[0]

    inter_distances = np.ones((total_labels, total_labels), dtype=np.float64)\
        * np.finfo(np.float64).max
    intra_distances = np.ones((total_labels, 1), dtype=np.float64)

    label_range = np.arange(0, len(unique_labels))

    for i in label_range:
        for j in label_range:
            if j == i:
                continue
            inter_distances[i, j] = delta_fast(
                (labels == unique_labels[i]), (labels == unique_labels[j]), distances
            )

        intra_distances[i] = intra_distance(
            (labels == unique_labels[i]), distances)

    divisor = np.max(intra_distances)

    if divisor == 0:
        return 0

    final_di = np.min(inter_distances) / divisor

    return float(final_di)


def dunn(distances: np.ndarray, labels: np.ndarray) -> float:
    return dunn_helper(distances, labels)


def dunn2(distances: np.ndarray, labels: np.ndarray) -> float:
    return dunn_helper(distances, labels, inter_distance2)


# @jit(nopython=True, cache=True)
def davies_bouldin_helper(
    distances: np.ndarray,
    labels: np.ndarray,
    delta_fast: Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        float
    ] = inter_distance
) -> float:
    unique_labels = np.unique(labels)
    total_labels = unique_labels.shape[0]

    label_range = np.arange(0, total_labels)

    inter_distances = np.zeros((total_labels, 1), dtype=np.float64)
    intra_distances = np.ones((total_labels, total_labels), dtype=np.float64) \
        * np.finfo(np.float64).min

    for i in label_range:
        inter_distances[i] = intra_distance(
            (labels == unique_labels[i]), distances)

    big_deltas = np.zeros((total_labels, 1), dtype=np.float64)

    for i in label_range:
        deltas = np.ones((total_labels - 1, 1), dtype=np.float64) \
            * np.finfo(np.float64).min
        idx = 0
        for j in label_range:
            if j == i:
                continue
            cached_intra_distance = max(
                intra_distances[i, j],  # type: ignore
                intra_distances[j, i]  # type: ignore
            )

            if cached_intra_distance == np.finfo(np.float64).min:
                cached_intra_distance = delta_fast(
                    (labels == unique_labels[i]),
                    (labels == unique_labels[j]),
                    distances
                )
                intra_distances[i, j] = cached_intra_distance
                intra_distances[j, i] = cached_intra_distance

            if cached_intra_distance == 0:
                deltas[idx] = 0
            else:
                deltas[idx] = (
                    inter_distances[i] + inter_distances[j]
                ) / cached_intra_distance
            idx += 1

        big_deltas[i] = np.max(deltas)

    return np.mean(big_deltas)


def davies_bouldin(distances: np.ndarray, labels: np.ndarray) -> float:
    return davies_bouldin_helper(distances, labels)


def davies_bouldin2(distances: np.ndarray, labels: np.ndarray) -> float:
    return davies_bouldin_helper(distances, labels, inter_distance2)


def davies_bouldin3(data: np.ndarray, labels: np.ndarray, distance_func: Callable) -> float:
    unique_labels = np.unique(labels)
    total_labels = unique_labels.shape[0]

    label_range = np.arange(0, total_labels)

    centroids = np.zeros(
        (unique_labels.shape[0], data.shape[1]), dtype=np.float64)
    intra_distances = np.zeros(unique_labels.shape[0], dtype=np.float64)

    for idx, label in enumerate(np.unique(labels)):
        cluster_i = data[labels == label]
        centroid = np.mean(cluster_i, axis=0, dtype=np.float64)
        centroids[idx] = centroid
        intra_distances[idx] = np.mean(distance_func(
            centroid, cluster_i), dtype=np.float64)

    centroid_inter_distance = distance_func(centroids, centroids)

    results = np.zeros(unique_labels.shape[0], dtype=np.float64)

    for i in label_range:
        ridx = 0
        iter_results = np.zeros(
            (unique_labels.shape[0] - 1, unique_labels.shape[0] - 1),
            dtype=np.float64
        )
        for j in label_range:
            if i == j:
                continue
            iter_results[ridx] = (
                intra_distances[i] + intra_distances[j]
            ) / centroid_inter_distance[i, j]
            ridx += 1
        results[i] = np.max(iter_results)
    return np.mean(results)


if __name__ == "__main__":
    import csv
    import os
    from pathlib import Path

    ruspini_path = os.path.join(
        Path(os.path.dirname(__file__)).parents[2],
        "shared",
        "datasets",
        "ruspini.csv"
    )

    with open(ruspini_path, "r+", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";", quoting=csv.QUOTE_NONE)
        ruspini_data = [np.array(line) for line in reader]
        column = ruspini_data.pop(0)
        ruspini_data = np.array(ruspini_data)

        ruspini_data, ruspini_labels = ruspini_data[:,
                                                    1:-1], ruspini_data[:, -1]

        ruspini_data = ruspini_data.astype(int)

        # ruspini_data = np.resize(ruspini_data, (5500, 2))
        # ruspini_labels = np.resize(ruspini_labels, (5500))

        distance = euclidean_distances(ruspini_data, ruspini_data)

        print("acc:", cluster_acc(distance, ruspini_labels, ruspini_labels))
        print("nmi: ", nmi(ruspini_labels, ruspini_labels))
        print("dunn1: ", dunn(distance, ruspini_labels))
        print("dunn2: ", dunn2(distance, ruspini_labels))
        print("davies_bouldin: ", davies_bouldin_helper(distance, ruspini_labels))
        print("davies_bouldin2: ", davies_bouldin2(distance, ruspini_labels))
        print("davies_bouldin3: ", davies_bouldin3(
            ruspini_data,
            ruspini_labels,
            euclidean_distances
        ))
        print("sklearn_davies_bouldin_score: ",
              davies_bouldin_score(ruspini_data, ruspini_labels))
        print("silhouette_score: ", silhouette(distance, ruspini_labels))
