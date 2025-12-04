# Embedding Generator

A Python script to generate text embeddings using various representation methods and store them as parquet files.

## Features

- Processes datasets with per-dataset configuration
- Each dataset can specify its own text column, delimiter, and representations
- Generates embeddings using multiple representation methods:
  - NGram
  - LIWC
  - MRC2
  - STagger
  - Word2Vec
  - Doc2Vec
  - FastText
  - SBert
- Saves embeddings as parquet files for efficient storage and retrieval
- Automatically cleans CUDA memory between iterations to prevent VRAM overflow
- Supports normalization of embeddings using MinMaxScaler
- Fully configurable via JSON configuration file

## Usage

### Basic Usage

```bash
# Using a configuration file (required)
python embedding_generator.py --config embedding_config.json

# With custom output folder and log level
python embedding_generator.py \
  --config embedding_config.json \
  --output-folder ../shared/embeddings \
  --log-level DEBUG
```

### Configuration File

Create a JSON configuration file (e.g., `embedding_config.json`):

```json
{
  "output_folder": "../shared/embeddings",
  "normalize_embeddings": true,
  "datasets": [
    {
      "path": "shared/datasets/news/file.csv",
      "text_column": "text",
      "delimiter": ",",
      "representations": [
        {
          "name": "NGram",
          "params": {
            "ngram_range": [1, 2],
            "max_features": 1000
          }
        },
        {
          "name": "SBert",
          "params": {
            "model_name": "paraphrase-MiniLM-L6-v2"
          }
        },
        {
          "name": "Word2Vec",
          "params": {
            "vector_size": 300,
            "window": 5,
            "min_count": 1,
            "workers": 4,
            "epochs": 10
          }
        }
      ]
    },
    {
      "path": "shared/datasets/reviews/data.csv",
      "text_column": "review_text",
      "delimiter": ",",
      "representations": [
        {
          "name": "SBert",
          "params": {
            "model_name": "all-MiniLM-L6-v2"
          }
        }
      ]
    }
  ]
}
```

### Configuration Options

**Global Options:**
- `output_folder`: Path where parquet files will be saved
- `normalize_embeddings`: Whether to normalize embeddings (default: true)
- `datasets`: Array of dataset configurations (required)

**Per-Dataset Options:**
- `path`: Path to the CSV file (can be relative or absolute)
- `text_column`: Name of the column containing text data (default: "text")
- `delimiter`: CSV delimiter (default: ",")
- `representations`: Array of representation configurations to generate for this dataset

**Per-Representation Options:**
- `name`: Name of the representation method (NGram, LIWC, MRC2, STagger, Word2Vec, Doc2Vec, FastText, SBert)
- `params`: Dictionary of parameters specific to the representation method

### Output Format

Embeddings are saved as parquet files with the naming convention:
```
{dataset_name}_{representation_name}.parquet
```

Each parquet file contains:
- Feature columns: `feat_0`, `feat_1`, ..., `feat_N`
- Metadata columns:
  - `dataset`: Name of the source dataset
  - `representation`: Name of the representation method
  - `meta_params`: Representation parameters used

### Reading Generated Embeddings

```python
import pandas as pd

# Read embeddings
df = pd.read_parquet("ruspini_Word2Vec.parquet")

# Get embedding matrix
embeddings = df[[col for col in df.columns if col.startswith("feat_")]].values

# Get metadata
dataset_name = df["dataset"].iloc[0]
representation = df["representation"].iloc[0]
```

## Memory Management

The script automatically:
- Processes one dataset at a time
- Cleans CUDA memory after each representation
- Runs garbage collection between iterations
- Uses `cuda.empty_cache()` and `cuda.synchronize()` to prevent VRAM overflow

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- torch
- pyarrow (for parquet support)
- All aiNet representation dependencies

## Command-Line Arguments

- `--config, -c`: Path to JSON configuration file (required)
- `--output-folder, -o`: Override output folder for parquet files (optional)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Example Workflow

1. Prepare your CSV files with text data
2. Create `embedding_config.json` with dataset paths and desired representations:
   ```json
   {
     "output_folder": "../shared/embeddings",
     "normalize_embeddings": true,
     "datasets": [
       {
         "path": "shared/datasets/news/file.csv",
         "text_column": "text",
         "representations": [
           {"name": "SBert", "params": {"model_name": "paraphrase-MiniLM-L6-v2"}},
           {"name": "Word2Vec", "params": {"vector_size": 300, "epochs": 10}}
         ]
       }
     ]
   }
   ```
3. Run the script:
   ```bash
   python embedding_generator.py --config embedding_config.json
   ```
4. Find generated embeddings in `../shared/embeddings/`
5. Load embeddings for analysis:
   ```python
   import pandas as pd
   embeddings = pd.read_parquet("../shared/embeddings/file_SBert.parquet")
   ```

## Error Handling

- Errors during representation generation are logged but don't stop the entire process
- Each representation is processed in a try-except block
- Memory is cleaned even if errors occur
- Failed representations are skipped, and processing continues

## Logging

The script provides detailed logging:
- INFO: High-level progress updates
- DEBUG: Detailed timing and parameter information
- ERROR: Exceptions and failures

Set the log level using `--log-level`:
```bash
python embedding_generator.py --log-level DEBUG
```
