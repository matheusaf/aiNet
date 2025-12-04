"""
Example script showing how to load and use the generated embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_embedding(parquet_path: str) -> tuple[np.ndarray, dict]:
    """
    Load embeddings from a parquet file.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        Tuple of (embedding_matrix, metadata)
    """
    # Read parquet file
    df = pd.read_parquet(parquet_path)

    # Extract embedding matrix (all columns starting with 'feat_')
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    embeddings = df[feature_cols].values

    # Extract metadata
    metadata = {
        "dataset": df["dataset"].iloc[0] if "dataset" in df.columns else None,
        "representation": df["representation"].iloc[0] if "representation" in df.columns else None,
        "shape": embeddings.shape,
        "num_samples": len(embeddings),
        "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
    }

    # Extract any additional metadata columns
    meta_cols = [col for col in df.columns if col.startswith("meta_")]
    for col in meta_cols:
        metadata[col.replace("meta_", "")] = df[col].iloc[0]

    return embeddings, metadata


def list_available_embeddings(embeddings_folder: str) -> pd.DataFrame:
    """
    List all available embeddings in a folder.

    Args:
        embeddings_folder: Path to embeddings folder

    Returns:
        DataFrame with information about available embeddings
    """
    parquet_files = list(Path(embeddings_folder).glob("*.parquet"))

    embeddings_info = []
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            embeddings_info.append(
                {
                    "file": parquet_file.name,
                    "dataset": df["dataset"].iloc[0] if "dataset" in df.columns else "unknown",
                    "representation": df["representation"].iloc[0]
                    if "representation" in df.columns
                    else "unknown",
                    "num_samples": len(df),
                    "embedding_dim": len([col for col in df.columns if col.startswith("feat_")]),
                    "path": str(parquet_file),
                }
            )
        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")

    return pd.DataFrame(embeddings_info)


def load_multiple_embeddings(
    embeddings_folder: str, dataset_name: str | None = None, representation: str | None = None
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Load multiple embeddings matching the criteria.

    Args:
        embeddings_folder: Path to embeddings folder
        dataset_name: Filter by dataset name (optional)
        representation: Filter by representation method (optional)

    Returns:
        Dictionary mapping file names to (embeddings, metadata) tuples
    """
    available = list_available_embeddings(embeddings_folder)

    # Apply filters
    if dataset_name:
        available = available[available["dataset"] == dataset_name]
    if representation:
        available = available[available["representation"] == representation]

    # Load embeddings
    result = {}
    for _, row in available.iterrows():
        embeddings, metadata = load_embedding(row["path"])
        result[row["file"]] = (embeddings, metadata)

    return result


def example_usage():
    """Example usage of the embedding loading functions."""
    embeddings_folder = "../shared/embeddings"

    print("=" * 80)
    print("Listing all available embeddings:")
    print("=" * 80)
    available = list_available_embeddings(embeddings_folder)
    print(available.to_string())
    print()

    if len(available) > 0:
        print("=" * 80)
        print("Loading first embedding:")
        print("=" * 80)
        first_file = available.iloc[0]["path"]
        embeddings, metadata = load_embedding(first_file)

        print(f"File: {first_file}")
        print(f"Dataset: {metadata['dataset']}")
        print(f"Representation: {metadata['representation']}")
        print(f"Shape: {metadata['shape']}")
        print(f"Number of samples: {metadata['num_samples']}")
        print(f"Embedding dimension: {metadata['embedding_dim']}")
        print(f"\nFirst 3 embeddings:\n{embeddings[:3]}")
        print()

        print("=" * 80)
        print("Loading all embeddings for a specific dataset:")
        print("=" * 80)
        dataset_name = metadata["dataset"]
        dataset_embeddings = load_multiple_embeddings(
            embeddings_folder, dataset_name=dataset_name
        )

        for file_name, (emb, meta) in dataset_embeddings.items():
            print(f"{file_name}: {meta['representation']} - {emb.shape}")
        print()

        print("=" * 80)
        print("Loading all embeddings for a specific representation:")
        print("=" * 80)
        repr_embeddings = load_multiple_embeddings(embeddings_folder, representation="SBert")

        for file_name, (emb, meta) in repr_embeddings.items():
            print(f"{file_name}: {meta['dataset']} - {emb.shape}")
        print()

    else:
        print("No embeddings found. Please run embedding_generator.py first.")


if __name__ == "__main__":
    example_usage()
