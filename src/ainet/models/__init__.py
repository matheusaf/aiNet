"""
	Module containing all immune models
"""


import sys

if "-m" not in sys.argv:
    from .ainet import AiNet
    from .sia import SIA

    __all__ = [
        "AiNet",
        "SIA"
    ]

__version__ = "0.5"
