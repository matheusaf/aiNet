# aiNet Clustering Algorithm

[![image](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/matheusaf/aiNet)
[![Pylint](https://github.com/matheusaf/aiNet/actions/workflows/pylint.yml/badge.svg)](https://github.com/matheusaf/aiNet/actions/workflows/pylint.yml)

Python library implementing the Artificial Immune Network (aiNet) clustering algorithm for text analysis. Designed for studying different text representations combined with immune-inspired clustering.

**Authors:** Matheus Amendoeira Ferraria
**Advisors:** Prof. Dr. Pedro Paulo Balbi de Oliveira, Prof. Dr. Leandro Nunes de Castro

## Installation

```bash
uv pip install https://github.com/matheusaf/aiNet.git
```

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