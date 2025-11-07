"""
    Module containg common functions using in this project
"""

from .distance_wrappers import cosine_distances, euclidean_distances
from .norma import norma
from .plot_dendogram import plot_dendrogram
from .progress_bar import print_progress_bar

__all__ = [
    "print_progress_bar",
    "norma",
    "euclidean_distances",
    "cosine_distances",
    "plot_dendrogram"
]

__version__ = "0.5"
