import numpy as np


def norma(matrix: np.ndarray) -> np.ndarray:
    """
                Function to normalize a matrix to range[0,1] using it's
                        min and max values
    """
    is_1d_array = False
    final_matrix = matrix.copy()

    if len(final_matrix.shape) == 1 or final_matrix.shape[0] == 1:
        final_matrix = final_matrix.reshape(-1, 1)
        is_1d_array = True

    n_columns = final_matrix.shape[1]

    normalized_matrix = final_matrix.copy().astype(float)

    for column_index in range(n_columns):
        column = final_matrix[:, column_index]

        min_value = column.min()
        max_value = column.max()
        lower = max_value - min_value

        if lower == 0:
            lower = 1

        normalized_matrix[:, column_index] = (column - min_value) / (lower)

    if is_1d_array:
        normalized_matrix = normalized_matrix.reshape(-1)

    return normalized_matrix
