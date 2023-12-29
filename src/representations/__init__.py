"""
    Module containg all represtations used in this project
"""


import sys

if "-m" not in sys.argv:
    from .representation import Representation
    from .sbert import SBert
    from .doc2vec import Doc2Vec
    from .liwc import LIWC
    from .mrc2 import MRC2
    from .ngram import NGram
    from .stagger import STagger
    from .word2vec import Word2Vec
    from .fasttext import FastText

    __all__ = [
        "Representation",
        "LIWC",
        "STagger",
        "MRC2",
        "NGram",
        "Word2Vec",
        "Doc2Vec",
        "SBert",
        "FastText"
    ]

__version__ = "1.0"
