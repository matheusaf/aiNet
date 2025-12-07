"""
Example script showing how to load and use the generated embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_embedding(parquet_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load embeddings from a parquet file.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        Tuple of (embedding_matrix, metadata)
    """
    # Read parquet file
    df = pd.read_parquet(parquet_path)

    # Extract embedding matrix (all columns starting with 'feat_')]
    feature_cols = [col for col in df.columns if not col.startswith("$meta_")]
    embeddings = df[feature_cols]

    # Extract metadata
    metadata = {
        "dataset": df["dataset"].iloc[0] if "dataset" in df.columns else None,
        "representation": df["representation"].iloc[0] if "representation" in df.columns else None,
        "shape": embeddings.shape,
        "num_samples": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
    }

    # Extract any additional metadata columns
    meta_cols = [col for col in df.columns if col.startswith("$meta_")]
    for col in meta_cols:
        metadata[col.replace("meta_", "")] = df[col].iloc[0]

    return embeddings, metadata


def main():
    """
    Example usage of loading embeddings.
    """
    # Path to the parquet file
    parquet_file = Path("../shared/embeddings/yelp_labelled_NGram.parquet")

    # Load embeddings
    embeddings, meta = load_embedding(str(parquet_file))

    embeddings.to_csv("loaded_embeddings.csv", sep=";", index=False)

    # Print metadata
    print("Metadata:")
    for key, value in meta.items():
        print(f"  {key}: {value}")

    # Print shape of embeddings
    print(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
