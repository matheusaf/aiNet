# aiNet Clustering Algorithm

[![image](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/matheusaf/aiNet)
[![Pylint](https://github.com/matheusaf/aiNet/actions/workflows/pylint.yml/badge.svg)](https://github.com/matheusaf/aiNet/actions/workflows/pylint.yml)

Python library implementing the Artificial Immune Network (aiNet) clustering algorithm for text analysis. Designed for studying different text representations combined with immune-inspired clustering.

**Authors:** Matheus Amendoeira Ferraria
**Advisors:** Prof. Dr. Pedro Paulo Balbi de Oliveira, Prof. Dr. Leandro Nunes de Castro

## Installation

### Using UV
```bash
uv add git+https://github.com/matheusaf/aiNet.git
```

### Using pip
```bash
pip install https://github.com/matheusaf/aiNet.git
```

It's currently compatatible with Python3.10 and Python3.11

## aiNet Model

Unsupervised clustering algorithm using artificial immune network principles. Builds an antibody population that learns to recognize patterns in data through clonal selection, mutation, and network pruning.

**Key Features:**
- Affinity-based antibody selection
- Clonal expansion with adaptive mutation
- Network suppression for redundancy removal
- MST-based cluster identification
- Multi-processing support
- Distance metrics: Euclidean, Cosine

**Main Parameters:**
- `number_of_antibodies`: Antibodies generated per iteration
- `clone_multiplier`: Clone population size
- `pruning_threshold`: Natural death threshold
- `suppression_threshold`: Network suppression threshold
- `hypermutation_rate`: Mutation/learning rate
- `max_iter`: Maximum iterations

## Representations

### Neural/Embedding-Based

#### SBert
Sentence-level embeddings using pre-trained Sentence Transformers models.

**Output:** Dense vectors (384-768 dimensions)
**Features:** Batch processing, normalization, stop word removal
**Use Case:** Semantic similarity tasks

#### Doc2Vec
Document-level embeddings using Gensim's PV-DM or PV-DBOW algorithms.

**Output:** Dense vectors (customizable dimension)
**Features:** Two training algorithms, bigram detection, multiprocessing
**Use Case:** Document clustering and classification

#### Word2Vec
Word-level embeddings (Skip-gram or CBOW) averaged at sentence level.

**Output:** Dense vectors (customizable dimension)
**Features:** Custom training or pre-trained models, bigram support
**Use Case:** General-purpose text representation

#### FastText
Character n-gram embeddings for handling rare/misspelled words.

**Output:** Dense vectors (customizable dimension)
**Features:** Subword information, robust to OOV words
**Use Case:** Noisy text, multiple languages, morphologically rich languages

### Linguistic/Dictionary-Based

#### LIWC
Linguistic Inquiry and Word Count - psycholinguistic features from dictionary lookup.

**Output:** Sparse vectors (70+ dimensions)
**Features:** Trie-based lookup, wildcard patterns
**Use Case:** Psychological/linguistic analysis, sentiment analysis

#### MRC2
Machine Readable Catalog - 43 linguistic and semantic word properties.

**Output:** Dense vectors (43 dimensions)
**Features:** Phonetic, frequency, semantic, morphological features
**Use Case:** Linguistic feature analysis, word complexity studies

#### STagger
Part-of-Speech tag distribution using Stanza POS tagger.

**Output:** Dense vectors (17 dimensions)
**Features:** GPU acceleration, parallel processing
**Use Case:** Syntactic structure analysis

### Statistical/Count-Based

#### NGram
N-gram based TF-IDF or count vectorization.

**Output:** Sparse/dense vectors (vocabulary size)
**Features:** Configurable n-gram range, stop word removal
**Use Case:** Baseline representation, language-agnostic tasks

## Usage

### Basic Example with Word2Vec

```python
from ainet.representations import Word2Vec
from ainet.models import AiNet
from sklearn.preprocessing import MinMaxScaler

# Initialize representation
representation = Word2Vec(
    train_corpus=documents,
    vector_size=100,
    stop_word_removal_enabled=True
)

# Generate vectors
features, vectors = representation.generate_representation(texts)

# Normalize to [0,1]
normalized_vectors = MinMaxScaler().fit_transform(vectors)

# Train aiNet
model = AiNet()
model.fit(
    normalized_vectors,
    number_of_antibodies=30,
    no_best_cells_taken_each_selection=4,
    clone_multiplier=5,
    pruning_threshold=0.8,
    max_iter=20
)

# Get cluster assignments
predictions = model.predict(normalized_vectors)
```

### Representation Usage Examples

#### SBert
```python
from ainet.representations import SBert

# Initialize with pre-trained model
sbert = SBert(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size=32,
    normalize_embeddings=False,
    stop_word_removal_enabled=True
)

# Generate representations
features, vectors = sbert.generate_representation(texts)
# Or as DataFrame
df = sbert.generate_representation(texts, as_dataframe=True)
```

#### Doc2Vec
```python
from ainet.representations import Doc2Vec

# Train new model
doc2vec = Doc2Vec(
    train_corpus=documents,
    vector_size=100,
    train_algorithm="PV-DBOW",  # or "PV-DM"
    window=5,
    min_count=2,
    epochs=40,
    use_bigrams=True,
    stop_word_removal_enabled=True
)

# Generate representations
features, vectors = doc2vec.generate_representation(texts)

# Or load pre-trained model
doc2vec = Doc2Vec.from_file("model.d2v")
```

#### FastText
```python
from ainet.representations import FastText

# Train new model
fasttext = FastText(
    train_corpus=documents,
    train_algorithm="skip-gram",  # or "cbow"
    vector_size=300,
    min_n=3,  # min character n-gram
    max_n=6,  # max character n-gram
    window=5,
    min_count=5,
    epochs=10,
    stop_word_removal_enabled=True
)

# Generate representations
features, vectors = fasttext.generate_representation(texts)

# Load Facebook pre-trained model
fasttext = FastText.from_facebook_model("path/to/cc.en.300.bin")
```

#### LIWC
```python
from ainet.representations import LIWC
from pathlib import Path

# Initialize with LIWC dictionary
liwc = LIWC(dic_filepath=Path("path/to/LIWC.dic"))

# Generate psycholinguistic features
features, vectors = liwc.generate_representation(texts)

# Features include word count and 70+ linguistic categories
df = liwc.generate_representation(texts, as_dataframe=True)
```

#### MRC2
```python
from ainet.representations import MRC2
from pathlib import Path

# Initialize with MRC dictionary (or load from pickle if exists)
mrc2 = MRC2(dic_filepath=Path("path/to/mrc2.dct"))

# Generate 43-dimensional linguistic feature vectors
features, vectors = mrc2.generate_representation(texts)

# Features include phonetic, frequency, semantic properties
df = mrc2.generate_representation(texts, as_dataframe=True)
```

#### STagger
```python
from ainet.representations import STagger

# Initialize with Stanza POS tagger
stagger = STagger(spacy_model_name="en_core_web_sm")

# Generate POS tag distributions (17 universal tags)
features, vectors = stagger.generate_representation(texts)

# Returns normalized frequency of each POS tag
df = stagger.generate_representation(texts, as_dataframe=True)
```

#### NGram
```python
from ainet.representations import NGram

# TF-IDF vectorization
ngram_tfidf = NGram(
    min_ngram_group=1,
    max_ngram_group=2,
    model_type="tf-idf",
    stop_words=None
)

# Count vectorization
ngram_count = NGram(
    min_ngram_group=1,
    max_ngram_group=3,
    model_type="count"
)

# Generate representations
features, vectors = ngram_tfidf.generate_representation(texts)
df = ngram_count.generate_representation(texts, as_dataframe=True)
```

## Representation Comparison

| Representation | Type | Typical Dims | Training | Strength |
|---|---|---|---|---|
| SBert | Neural | 384-768 | Pre-trained | Semantic similarity |
| Word2Vec | Neural | 50-300 | Required | Word meaning capture |
| Doc2Vec | Neural | 50-300 | Required | Document semantics |
| FastText | Neural | 100-300 | Required | Robust to misspellings |
| LIWC | Dictionary | 70+ | None | Psychological insights |
| MRC2 | Dictionary | 43 | None | Word properties |
| STagger | POS | 17 | Pre-trained | Syntactic structure |
| NGram | Statistical | Variable | Optional | Language-agnostic |