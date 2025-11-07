from os import cpu_count

from numpy import any as np_any
from numpy import array, fill_diagonal, isnan, nan_to_num, ndarray
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist


def euclidean_distances(
    arr1: ndarray,
    arr2: ndarray,
    normalize=False,
    using_multiprocessing=False
) -> ndarray:
    """
                Sklearn euclidean_distances wrapper
    """
    if arr1.size == 0 or arr2.size == 0:
        return array([])

    if len(arr1.shape) == 1 or arr1.shape[0] == 1:
        arr1 = arr1.reshape(1, -1)

    if len(arr2.shape) == 1 or arr2.shape[0] == 1:
        arr2 = arr2.reshape(1, -1)

    if arr1.size > 8000 or arr2.size > 8000:
        dist = pairwise_distances(
            arr1,
            arr2,
            metric="euclidean",
            n_jobs=cpu_count() if not using_multiprocessing else 1
        )
    else:
        dist = cdist(
            arr1,
            arr2,
            metric="euclidean"
        )

    if normalize and (dist.max() > 1 or dist.min() < 0):
        # return normalize(dist, norm="l2")
        return dist.clip(0, 1)
    return dist


def cosine_distances(
    arr1: ndarray,
    arr2: ndarray,
    normalize=True,
    using_multiprocessing=False
) -> ndarray:
    """
        Sklearn cosine distance wrapper
    """

    if arr1.size == 0 or arr2.size == 0:
        return array([])

    if len(arr1.shape) == 1 or arr1.shape[0] == 1:
        arr1 = arr1.reshape(1, -1)

    if len(arr2.shape) == 1 or arr2.shape[0] == 1:
        arr2 = arr2.reshape(1, -1)

    if arr1.size > 100_000 or arr2.size > 100_000:
        dist = pairwise_distances(
            arr1,
            arr2,
            metric="cosine",
            n_jobs=cpu_count() if not using_multiprocessing else 1
        )
    else:
        dist = cdist(
            arr1,
            arr2,
            metric="cosine",
        )

    if np_any(
        isnan(dist)
    ):
        dist = nan_to_num(dist, nan=1.0)

    n, m = dist.shape
    if n == m and np_any(dist > 0):
        fill_diagonal(dist, 0)

    return dist
