"""
    Word2Vec Model
"""
from multiprocessing import get_context
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import gensim.downloader
import numpy as np
import spacy
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.phrases import Phrases
from gensim.models.word2vec import Word2Vec as GensimWord2Vec
from pandas import DataFrame
from spacy.lang.en.stop_words import STOP_WORDS

from utils.text.processing import remove_punctuations

from .representation import Representation


class DocIter:
    """
    Generator class to stream data from files
    """

    def __init__(self, document_path) -> None:
        self.document_path = document_path

    def __iter__(self) -> Iterator[str]:  # type: ignore
        with open(self.document_path, "r", encoding="utf-8") as file:
            for line in file:
                yield line

    def __next__(self) -> str:
        return next(self.__iter__())


def init_worker(model: KeyedVectors, features: np.ndarray) -> None:
    global global_model, global_features

    global_model = model
    global_features = features


def generate_representation_helper(sentence: str) -> np.ndarray:
    global global_model, global_features
    sentence_result = np.zeros(global_features.shape[0], dtype=np.float32)

    tokens: list[str] = sentence.split()

    if len(tokens) == 0:
        return sentence_result
        
    for token in tokens:
        try:
            predict_result = global_model[token]
            # concat word vector to array
            sentence_result += predict_result
        except KeyError:
            continue

    # divide by number of ALL tokens to get average
    sentence_result = sentence_result / np.float32(len(tokens))

    return sentence_result


class Word2Vec(Representation):
    """
    The following methods are used to access the lexicon and category names.
    sg ({1,0}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
    """

    __slots__ = ["__model_"]

    __model_: KeyedVectors

    def __init__(
        self,
        document_path: Optional[Path | str] = None,
        vector_size: int = 50,
        train_algorithm: str = "cbow",
        window: int = 5,
        min_count: int = 2,
        epochs: int = 40,
        workers: int = os.cpu_count() or 1,
        negative: int = 5,
        train_corpus: Optional[list[str]] = None,
        trained_model: Optional[Any] = None,
        use_bigrams: Optional[bool] = None,
        bigram_min_count: int = 20,
        bigram_progress_per: int = 10000,
        spacy_model_name: str = "en_core_web_sm",
        stop_word_removal_enabled: bool = True,
        stop_words: Optional[set[str]] = None,  # type: ignore
    ) -> None:
        super().__init__(stop_word_removal_enabled=stop_word_removal_enabled)
        self.features = np.arange(vector_size)
        self._representation = np.array([])

        if stop_word_removal_enabled:
            self._spacy_model = spacy.load(
                spacy_model_name, disable=["parser", "ner", "textcat", "taggers"]
            )

            self._stop_words = (
                stop_words if stop_words is not None and len(stop_words) > 0 else STOP_WORDS
            )  # type: ignore

        if trained_model:
            if isinstance(trained_model, KeyedVectors):
                self.__model_ = trained_model
                self.features = np.arange(trained_model.vector_size)
                return

            raise TypeError("model is not a valid type")

        assert train_corpus or document_path, "train_corpus or document_path must be provided"

        valid_train_algorithms = ["cbow", "skip-gram"]

        assert (
            train_algorithm.casefold().strip() in valid_train_algorithms
        ), f"train_algorithm must be in {valid_train_algorithms}"

        gensim_model = GensimWord2Vec(
            vector_size=vector_size,
            sg=1 if train_algorithm.casefold().strip() == "skip-gram" else 0,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers,
            negative=negative,
        )

        self.__train_model_(
            gensim_model,
            document_path=document_path,
            train_corpus=train_corpus,
            use_bigrams=use_bigrams,
            bigram_min_count=bigram_min_count,
            bigram_progress_per=bigram_progress_per,
        )

        self.__model_ = gensim_model.wv

    def __getstate__(self) -> dict:
        obj_dict = {
            slot: getattr(self, slot)
            for slot in [*super().__slots__, *self.__slots__]
            if slot not in {"_stop_words", "_spacy_model"}
        }

        return dict(obj_dict)

    def __setstate__(self, state: dict) -> None:
        for slot, value in state.items():
            object.__setattr__(self, slot, value)

    def __train_model_(
        self,
        gensim_model: GensimWord2Vec,
        document_path: Path | str | None,
        train_corpus: list[str] | None,
        use_bigrams: bool | None,
        bigram_min_count: int,
        bigram_progress_per: int,
    ) -> None:
        corpus_iterable = []

        if train_corpus:
            corpus_iterable = train_corpus
        else:
            corpus_iterable = list(DocIter(document_path))

        corpus_iterable = [sentence.split() for sentence in self.pre_process(corpus_iterable)]

        if use_bigrams:
            bigram = Phrases(
                corpus_iterable, min_count=bigram_min_count, progress_per=bigram_progress_per
            )
            frozen_bigram = bigram.freeze()
            corpus_iterable = frozen_bigram[corpus_iterable]

        gensim_model.build_vocab(corpus_iterable)
        gensim_model.train(
            corpus_iterable, total_examples=gensim_model.corpus_count, epochs=gensim_model.epochs
        )

    @classmethod
    def from_file(cls, model_filename: str, stop_word_removal_enabled=True) -> "Word2Vec":
        """
        Restore object from file
        """
        if os.path.exists(model_filename):
            return cls(
                trained_model=GensimWord2Vec.load(model_filename),
                stop_word_removal_enabled=stop_word_removal_enabled,
            )
        raise FileNotFoundError(model_filename)

    @classmethod
    def from_trained_model(cls, model_name: str, stop_word_removal_enabled=True) -> "Word2Vec":
        """
        Load object from trained model
        """
        if model_name in dict(gensim.downloader.info()["models"]).keys():
            return cls(
                trained_model=gensim.downloader.load(model_name),
                stop_word_removal_enabled=stop_word_removal_enabled,
            )
        raise NameError(f"model '{model_name}' not found")

    def pre_process(self, sentences: list[str]) -> Iterator[str]:
        processed_sentences = list(super().pre_process(sentences))

        processed_sentences = self.remove_stop_words(processed_sentences)

        for sentence in processed_sentences:
            yield sentence

    def _pre_process_sentence(self, sentence: str) -> str:
        processed_sentence = super()._pre_process_sentence(sentence)
        return remove_punctuations(processed_sentence)

    def generate_representation(
        self, sentences: list[str], as_dataframe: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        mp_context = get_context("fork")

        n_processes = (os.cpu_count() or 1) if len(sentences) > 1_000 else 1
        chunksize = len(sentences) // n_processes

        processed_sentences = self.pre_process(sentences)

        with mp_context.Pool(
            processes=n_processes, initializer=init_worker, initargs=(self.__model_, self.features)
        ) as pool:
            results = pool.imap(
                generate_representation_helper,
                processed_sentences,
                chunksize=chunksize,
            )
            self._representation = np.array(
                list(results),
                dtype=np.float32,
            )

        if as_dataframe:
            return DataFrame(data=self._representation, columns=self._features)

        return (self.features, self.representation)


Representation.register(Word2Vec)

if __name__ == "__main__":
    spacy.prefer_gpu()  # type: ignore
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

    bow = Word2Vec.from_trained_model("word2vec-google-news-300", stop_word_removal_enabled=True)

    # bow = Word2Vec(
    #     train_corpus=TRAIN_CORPUS,  # type: ignore
    #     stop_word_removal_enabled=True,
    #     train_algorithm="cbow",
    # )  # type: ignore
    df = bow.generate_representation(
        [*TRAIN_CORPUS, *TRAIN_CORPUS, *TRAIN_CORPUS], as_dataframe=True
    )

    print(df)
