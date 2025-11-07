"""
    Doc2Vec Model
"""
import os
from collections.abc import Generator, Iterator
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional

import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec as GensimDoc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phrases
from pandas import DataFrame
from spacy.lang.en.stop_words import STOP_WORDS

from utils.text.processing import remove_punctuations

from .representation import Representation

# PV-DM is analogous to Word2Vec CBOW. The doc-vectors are obtained by training
# a neural network on the synthetictask of predicting a center word
# based an average of both context word-vectors and the full document’s doc-vector.

# from gensim.test.utils import get_tmpfile
# fname = get_tmpfile("my_doc2vec_model")
# model.save(fname)
# model = Doc2Vec.load(fname)  # you can co ntinue training with the loaded model!


class TaggedDocumentIter:
    """
    Generator class to stream data from files
    """

    def __init__(
        self, document_path: str, tagged_doc: bool | None = True, bigram: Any | None = None
    ) -> None:
        self.document_path = document_path
        self.bigram = bigram
        self.tagged_doc = tagged_doc

    def __iter__(self) -> Generator[str]:  # type: ignore
        with open(self.document_path, "r", encoding="utf-8") as file:
            for line in file:
                yield line

    def __next__(self) -> str:
        return next(self.__iter__())


def init_worker(model: GensimDoc2Vec) -> None:
    global global_model

    global_model = model


def generate_representation_helper(sentence: str) -> np.ndarray:
    global global_model
    tokens = sentence.split()
    return np.array(global_model.infer_vector(tokens), dtype=np.float32)


class Doc2Vec(Representation):  # type: ignore
    """
    The following methods are used to access the lexicon and category names.
    dm ({1,0}, optional) – Defines the training algorithm.
        If dm=1, ‘distributed memory’ (PV-DM) is used.
        Otherwise, distributed bag of words (PV-DBOW) is employed.
    """

    __slots__ = ["__model_"]

    __model_: GensimDoc2Vec

    def __init__(
        self,
        document_path: Path | str | None = None,
        vector_size: int = 50,
        train_algorithm: str = "PV-DBOW",
        window: int = 5,
        min_count: int = 2,
        epochs: int = 40,
        workers: int = 4,
        negative: int = 5,
        train_corpus: list[str] | None = None,
        trained_model: Any | None = None,
        use_bigrams: bool | None = None,
        bigram_min_count: int = 20,
        bigram_progress_per: int = 10000,
        spacy_model: str = "en_core_web_sm",
        stop_word_removal_enabled: bool = True,
        stop_words: Optional[set[str]] = None,  # type: ignore
    ) -> None:
        super().__init__(stop_word_removal_enabled=stop_word_removal_enabled)
        self.features = np.arange(vector_size)
        self._representation = np.array([])

        if stop_word_removal_enabled:
            self._spacy_model = spacy.load(
                spacy_model, disable=["parser", "ner", "textcat", "taggers"]
            )

            self._stop_words = (
                stop_words if stop_words is not None and len(stop_words) > 0 else STOP_WORDS
            )  # type: ignore

        if trained_model:
            if isinstance(trained_model, GensimDoc2Vec):
                self.__model_ = trained_model
                return

            raise TypeError("model is not a valid type")

        assert train_corpus or document_path, "train_corpus or document_path must be provided"

        valid_train_algorithms = ["pv-dbow", "pv-dm"]

        assert (
            train_algorithm.casefold().strip() in valid_train_algorithms
        ), f"train_algorithm must be in {valid_train_algorithms}"

        self.__model_ = GensimDoc2Vec(
            vector_size=vector_size,
            dm=1 if train_algorithm.casefold().strip() == "pv-dm" else 0,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers,
            negative=negative,
        )

        self.__train_model_(
            train_corpus=train_corpus,
            document_path=document_path,
            use_bigrams=use_bigrams,
            bigram_min_count=bigram_min_count,
            bigram_progress_per=bigram_progress_per,
        )

    def __train_model_(
        self,
        train_corpus: list[str] | None,
        document_path: Path | str | None,
        use_bigrams: bool | None,
        bigram_min_count: int = 20,
        bigram_progress_per: int = 10000,
    ) -> None:
        corpus_iterable = []

        if train_corpus:
            corpus_iterable = train_corpus
        else:
            corpus_iterable = list(TaggedDocumentIter(document_path))  # type: ignore

        corpus_iterable = [sentence.split() for sentence in self.pre_process(corpus_iterable)]

        if use_bigrams:
            bigram = Phrases(
                corpus_iterable, min_count=bigram_min_count, progress_per=bigram_progress_per
            )
            frozen_bigram = bigram.freeze()
            corpus_iterable = frozen_bigram[corpus_iterable]

        corpus_iterable = [
            TaggedDocument(words=sentence, tags=[idx])
            for idx, sentence in enumerate(corpus_iterable)
        ]

        self.__model_.build_vocab(corpus_iterable=corpus_iterable)
        self.__model_.train(
            corpus_iterable, total_examples=self.__model_.corpus_count, epochs=self.__model_.epochs
        )

    @classmethod
    def from_file(cls, model_filename: str, stop_word_removal_enabled=True) -> "Doc2Vec":
        """
        Abstract Constructor to Create model from file
        """
        if os.path.exists(model_filename):
            return cls(
                trained_model=GensimDoc2Vec.load(model_filename, mmap="r"),
                stop_word_removal_enabled=stop_word_removal_enabled,
            )
        raise FileNotFoundError(f"model file '{model_filename}' not found")

    def pre_process(self, sentences: list[str]) -> Iterator[str]:
        processed_sentences = list(super().pre_process(sentences))

        processed_sentences = self.remove_stop_words(processed_sentences)

        for processed_sentence in processed_sentences:
            yield processed_sentence

    def _pre_process_sentence(self, sentence: str) -> str:
        processed_sentence = super()._pre_process_sentence(sentence)
        return remove_punctuations(processed_sentence)

    def generate_representation(
        self, sentences: list[str], as_dataframe: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        mp_context = get_context("fork")

        n_processes = (os.cpu_count() or 1) if len(sentences) > 2_500 else 1
        chunksize = len(sentences) // n_processes

        processed_sentences = self.pre_process(sentences)

        with mp_context.Pool(
            processes=n_processes, initializer=init_worker, initargs=(self.__model_,)
        ) as pool:
            results = pool.imap(
                generate_representation_helper, processed_sentences, chunksize=chunksize
            )
            self._representation = np.array(
                list(results),
                dtype=np.float32,
            )

        if as_dataframe:
            return DataFrame(data=self._representation, columns=self._features)

        return (self.features, self.representation)


Representation.register(Doc2Vec)

if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # from sklearn.manifold import TSNE

    TRAIN_CORPUS = [
        "Use this metaclass to create an ABC",
        "An ABC can be subclassed directly, and then acts as a mix-in class.",
        "You can also register unrelated concrete classes",
        "(even built-in  classes) and unrelated ABCs as ",
        "virtual subclasses” – these and their descendants will be considered",
        "subclasses of the registering ABC by the built",
    ] * 250

    bow = Doc2Vec(train_corpus=TRAIN_CORPUS, stop_word_removal_enabled=True)
    df = bow.generate_representation(
        [*TRAIN_CORPUS, *TRAIN_CORPUS, *TRAIN_CORPUS], as_dataframe=True
    )
    print(df)