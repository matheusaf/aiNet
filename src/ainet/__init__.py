"""
aiNet - Artificial Immune Network

A Python package implementing Artificial Immune Network algorithms
for clustering and data analysis.
"""

# Read version from pyproject.toml
try:
    from importlib.metadata import version
    __version__ = version("aiNet")
except Exception:
    __version__ = "0.1.1"  # Fallback version

# Define submodules available for import
__all__ = ["models", "representations", "utils"]
