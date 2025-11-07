import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(mst_model: AgglomerativeClustering, **kwargs) -> None:
    """
                Create linkage matrix and then plot the dendrogram
        create the counts of samples under each node
    """

    counts = np.zeros(mst_model.children_.shape[0])
    n_samples = len(mst_model.labels_)
    for i, merge in enumerate(mst_model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [mst_model.children_, mst_model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
