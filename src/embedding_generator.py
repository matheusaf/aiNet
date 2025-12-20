"""
Embedding Generator Script

This script generates text embeddings using various representation methods
and stores them as parquet files for later use.
"""

import sys
import csv
import gc
import json
import logging as lg
import os
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import cuda

import ainet.representations as representation_models
from ainet.utils import print_progress_bar


class EmbeddingGenerator:
    """Generates and stores text embeddings for various representation methods."""

    __slots__ = ["__logger_", "__config_"]

    # Representation methods that should be normalized
    __normed_representations_ = {
        "NGram": True,
        "LIWC": True,
        "MRC2": True,
        "STagger": True,
        "Word2Vec": True,
        "Doc2Vec": True,
        "FastText": True,
        "SBert": True,
    }

    # Available representation methods
    __available_representations_ = [
        "NGram",
        "LIWC",
        "MRC2",
        "STagger",
        "Word2Vec",
        "Doc2Vec",
        "FastText",
        "SBert",
    ]

    def __init__(self, config_path: str | None = None, logger: lg.Logger | None = None) -> None:
        """
        Initialize the embedding generator.

        Args:
            config_path: Path to JSON configuration file
            logger: Logger instance
        """
        init_start = time()

        # Setup logger
        if logger is None:
            self.__logger_ = self.__setup_logger_()
        else:
            self.__logger_ = logger

        # Load configuration
        self.__config_ = self.__load_config_(config_path)

        self.__logger_.debug(
            "Instantiating %s took %0.5f second(s)", self.__class__.__name__, time() - init_start
        )

    def __setup_logger_(self) -> lg.Logger:
        """Setup default logger with file and console handlers."""
        logger = lg.getLogger("root")
        logger.setLevel(lg.DEBUG)

        # File handler
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logs",
        )
        os.makedirs(log_dir, exist_ok=True)

        file_logger = lg.FileHandler(
            filename=os.path.join(
                log_dir,
                f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            ),
            encoding="utf-8",
        )
        file_logger.setLevel(lg.DEBUG)
        file_logger.setFormatter(
            lg.Formatter(
                "(%(asctime)s)[%(levelname)s:%(name)s] "
                "%(module)s.%(filename)s.%(funcName)s => | %(message)s |"
            )
        )
        file_logger.addFilter(lg.Filter(name="root"))

        # Console handler
        console_handler = lg.StreamHandler()
        console_handler.setLevel(lg.INFO)
        console_handler.setFormatter(
            lg.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        logger.addHandler(file_logger)
        logger.addHandler(console_handler)

        return logger

    def __load_config_(self, config_path: str | None) -> dict:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Default configuration
        return {
            "datasets": [],
            "output_folder": os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "shared", "embeddings"
            ),
            "normalize_embeddings": True,
        }

    def __read_csv_file_(
        self, dataset_path: str, text_column: str, delimiter: str
    ) -> pd.DataFrame:
        """
        Read CSV file into DataFrame.

        Args:
            dataset_path: Path to dataset file
            text_column: Name of text column
            delimiter: CSV delimiter

        Returns:
            DataFrame containing the dataset
        """
        assert dataset_path and os.path.exists(dataset_path), f"File '{dataset_path}' does not exist"
        assert text_column, "text_column cannot be None or empty"

        read_start = time()
        csv.field_size_limit(sys.maxsize)

        with open(dataset_path, "r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL, quotechar='"')
            dataset_data = [row for row in reader]

        assert dataset_data and len(dataset_data) > 0, "Failed to read file"

        self.__logger_.info(
            "Read file '%s' with %d line(s) in %0.5f second(s)",
            dataset_path,
            len(dataset_data),
            time() - read_start,
        )

        dataset_header = dataset_data.pop(0)
        dataset = pd.DataFrame(columns=dataset_header, data=dataset_data)

        # Ensure text column exists
        if text_column not in dataset.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")

        return dataset

    def __instantiate_representation_(
        self,
        representation_name: str,
        representation_params: dict,
        dataset_df: pd.DataFrame,
        text_column: str,
    ) -> Any:
        """
        Instantiate a representation model.

        Args:
            representation_name: Name of the representation method
            representation_params: Parameters for the representation
            dataset_df: Dataset DataFrame
            text_column: Name of text column

        Returns:
            Instantiated representation model
        """
        params = representation_params.copy()

        # For models that need training corpus
        if representation_name in {"Word2Vec", "Doc2Vec", "FastText"} and not params.get(
            "trained_model", False
        ):
            if "train_corpus" not in params:
                params["train_corpus"] = dataset_df[text_column].tolist()

        self.__logger_.debug(
            "Instantiating representation model '%s' with params %s",
            representation_name,
            {**params, "train_corpus": bool(params.get("train_corpus", False))},
        )

        # Handle pre-trained models
        if trained_model := params.get("trained_model", False):
            if representation_name == "Word2Vec":
                return representation_models.Word2Vec.from_trained_model(trained_model)
            if representation_name == "FastText":
                return representation_models.FastText.from_facebook_model(trained_model)

        # Instantiate new models
        if representation_name == "NGram":
            return representation_models.NGram(**params)
        if representation_name == "LIWC":
            return representation_models.LIWC(**params)
        if representation_name == "STagger":
            return representation_models.STagger(**params)
        if representation_name == "MRC2":
            return representation_models.MRC2(**params)
        if representation_name == "Word2Vec":
            return representation_models.Word2Vec(**params)
        if representation_name == "Doc2Vec":
            return representation_models.Doc2Vec(**params)
        if representation_name == "FastText":
            return representation_models.FastText(**params)
        if representation_name == "SBert":
            return representation_models.SBert(**params)

        raise ValueError(f"Unknown representation: {representation_name}")

    def __generate_embedding_(
        self,
        representation_name: str,
        representation_params: dict,
        dataset_df: pd.DataFrame,
        text_column: str,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for a dataset using a specific representation.

        Args:
            representation_name: Name of the representation method
            representation_params: Parameters for the representation
            dataset_df: Dataset DataFrame
            text_column: Name of text column
            normalize: Whether to normalize the embeddings

        Returns:
            Tuple of (features, embeddings)
        """
        gen_start = time()

        # Clean CUDA cache before generating
        cuda.empty_cache()
        self.__logger_.debug("Cleaned CUDA cache in %0.5f second(s)", time() - gen_start)

        # Instantiate representation model
        representation_model = self.__instantiate_representation_(
            representation_name, representation_params, dataset_df, text_column
        )

        # Generate representation
        features, embeddings = representation_model.generate_representation(
            dataset_df[text_column].tolist()
        )

        self.__logger_.info(
            "Generated representation '%s' with shape (%d, %d) in %0.5f second(s)",
            representation_name,
            embeddings.shape[0],
            embeddings.shape[1],
            time() - gen_start,
        )

        # Normalize if needed
        if normalize and self.__normed_representations_.get(representation_name, False):
            norm_start = time()
            embeddings = MinMaxScaler().fit_transform(embeddings.astype(np.float64))
            self.__logger_.debug(
                "Normalized representation '%s' in %0.5f second(s)",
                representation_name,
                time() - norm_start,
            )

        return features, embeddings

    def __save_embeddings_(
        self,
        embeddings: np.ndarray,
        features: np.ndarray,
        dataset_name: str,
        representation_name: str,
        export_name :str,
        output_folder: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Save embeddings to parquet file.

        Args:
            embeddings: Embedding matrix
            features: Feature names
            dataset_name: Name of the dataset
            representation_name: Name of the representation method
            output_folder: Output folder path
            metadata: Optional metadata to include
        """
        save_start = time()

        final_export_name = export_name if export_name is not None else representation_name

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, dataset_name), exist_ok=True)

        # Ensure feature names are strings (required for parquet)
        string_features = np.array(features).astype(str).tolist()

        # Create DataFrame from embeddings
        df = pd.DataFrame(embeddings, columns=string_features)

        # Add metadata columns
        df["$dataset"] = dataset_name
        df["$representation"] = representation_name

        if metadata:
            for key, value in metadata.items():
                df[f"$meta_{key}"] = str(value)

        # Save to parquet
        output_path = os.path.join(
            output_folder, dataset_name, f"{final_export_name}.parquet"
        )
        df.to_parquet(output_path, index=True, compression="snappy")

        self.__logger_.info(
            "Saved embeddings to '%s' in %0.5f second(s)", output_path, time() - save_start
        )

    def __clean_memory_(self) -> None:
        """Clean up memory and CUDA cache."""
        clean_start = time()

        # Run garbage collection multiple times to ensure cleanup
        gc.collect()
        gc.collect()

        # Clear CUDA cache
        # if cuda.is_available():
            # cuda.empty_cache()
            # cuda.synchronize()

        self.__logger_.debug(
            "Cleaned memory and CUDA cache in %0.5f second(s)", time() - clean_start
        )

    def generate_all_embeddings(self) -> None:
        """
        Generate embeddings for all datasets and representations.

        This method loops through all datasets defined in the config
        and generates embeddings for each representation method specified
        for that dataset.
        """
        output_folder = self.__config_["output_folder"]
        normalize = self.__config_.get("normalize_embeddings", True)
        datasets_config = self.__config_.get("datasets", [])

        total_datasets = len(datasets_config)

        if total_datasets == 0:
            self.__logger_.warning("No datasets found in configuration")
            return

        self.__logger_.info("Processing %d dataset(s)", total_datasets)

        # Process each dataset
        for dataset_idx, dataset_config in enumerate(datasets_config, start=1):
            dataset_start = time()

            # Get dataset configuration
            dataset_path = dataset_config.get("path")
            if not dataset_path:
                self.__logger_.error("Dataset %d missing 'path' field, skipping", dataset_idx)
                continue

            # Convert relative paths to absolute
            if not os.path.isabs(dataset_path):
                base_dir = os.path.dirname(__file__)
                dataset_path = os.path.join(base_dir, dataset_path)

            # Get dataset-specific settings with defaults
            text_column = dataset_config.get("text_column", "text")
            delimiter = dataset_config.get("delimiter", ",")
            representations = dataset_config.get("representations", [])

            dataset_name = dataset_config.get(
                "dataset_name",
                Path(dataset_path).stem
            )
            total_representations = len(representations)

            self.__logger_.info(
                "\n========== Processing dataset %d/%d: %s ==========",
                dataset_idx,
                total_datasets,
                dataset_name,
            )
            self.__logger_.info(
                "  Dataset path: %s", dataset_path
            )
            self.__logger_.info(
                "  Representations to generate: %d", total_representations
            )

            try:
                # Read dataset
                dataset_df = self.__read_csv_file_(dataset_path, text_column, delimiter)

                # Process each representation for this dataset
                for repr_idx, repr_config in enumerate(representations, start=1):
                    repr_start = time()

                    repr_name = repr_config.get("name")
                    if not repr_name:
                        self.__logger_.error(
                            "  Representation %d missing 'name' field, skipping", repr_idx
                        )
                        continue

                    repr_export_name = repr_config.get("export_path", None)

                    repr_params = repr_config.get("params", {})

                    self.__logger_.info(
                        "  --- Representation %d/%d: %s ---",
                        repr_idx,
                        total_representations,
                        repr_name,
                    )

                    try:
                        # Generate embeddings
                        features, embeddings = self.__generate_embedding_(
                            representation_name=repr_name,
                            representation_params=repr_params,
                            dataset_df=dataset_df,
                            text_column=text_column,
                            normalize=normalize,
                        )

                        # Save embeddings
                        self.__save_embeddings_(
                            embeddings=embeddings,
                            features=features,
                            dataset_name=dataset_name,
                            representation_name=repr_name,
                            export_name=repr_export_name,
                            output_folder=output_folder,
                            metadata={"params": repr_params},
                        )

                    except Exception as repr_error:
                        self.__logger_.error(
                            "  ERROR processing representation '%s': %s",
                            repr_name,
                            str(repr_error),
                            exc_info=True,
                        )

                    finally:
                        # Clean memory after each representation
                        self.__clean_memory_()

                        # Display progress
                        repr_time = time() - repr_start
                        print_progress_bar(
                            repr_idx,
                            total_representations,
                            f"Dataset: {dataset_name} - Representation: {repr_name}",
                            f"Last representation took {repr_time:0.5f} second(s)",
                        )

            except Exception as dataset_error:
                self.__logger_.error(
                    "ERROR processing dataset '%s': %s",
                    dataset_name,
                    str(dataset_error),
                    exc_info=True,
                )

            finally:
                # Clean memory after each dataset
                self.__clean_memory_()

                # Display progress
                dataset_time = time() - dataset_start
                print_progress_bar(
                    dataset_idx,
                    total_datasets,
                    f"Processing dataset {dataset_idx}/{total_datasets}: {dataset_name}",
                    f"Last dataset took {dataset_time:0.5f} second(s)",
                )

        self.__logger_.info("\n========== Embedding generation complete! ==========")


def main():
    """Main entry point."""
    import spacy
    
    spacy.require_cpu()
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate text embeddings and save as parquet files"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to JSON configuration file (required)",
    )
    # parser.add_argument(
    #     "--output-folder", "-o", type=str, default=None, help="Override output folder for parquet files"
    # )
    # parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logger
    # logger = lg.getLogger(__name__)
    # logger.setLevel(getattr(lg, args.log_level.upper()))

    # console_handler = lg.StreamHandler()
    # console_handler.setLevel(getattr(lg, args.log_level.upper()))
    # formatter = lg.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # Create generator
    generator = EmbeddingGenerator(config_path=args.config)

    # Override config with command-line arguments if provided
    # if args.output_folder:
    #     generator._EmbeddingGenerator__config_["output_folder"] = args.output_folder

    # Generate embeddings
    generator.generate_all_embeddings()


if __name__ == "__main__":
    main()
