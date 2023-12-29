import os
from collections.abc import Generator, Iterator
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional

import numpy as np
import spacy
from gensim.models.fasttext import FastText as GensimFastText
from gensim.models.fasttext import load_facebook_model
from gensim.test.utils import datapath
from pandas import DataFrame
from spacy.lang.en.stop_words import STOP_WORDS

from utils.text.processing import remove_punctuations

from .representation import Representation


class FileIter:
    """
    Generator class to stream data from files
    """

    def __init__(self, document_path: Path | str) -> None:
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"file '{document_path}' not found")

        self.document_path = datapath(document_path)

    def __iter__(self) -> Generator[str]:  # type: ignore
        with open(self.document_path, "r", encoding="utf-8") as file:
            for line in file:
                yield line

    def __next__(self) -> str:
        return next(self.__iter__())


def init_worker(model: GensimFastText, features: np.ndarray) -> None:
    global global_model, global_features

    global_model = model
    global_features = features


def generate_representation_helper(sentence: str) -> np.ndarray:
    global global_model, global_features
    sentence_result = np.zeros(global_features.shape[0], dtype=np.float32)

    tokens = sentence.split()

    if len(tokens) == 0:
        return sentence_result

    for token in tokens:
        sentence_result += global_model.wv[token]

    sentence_result = sentence_result / np.float32(len(tokens))

    return sentence_result


class FastText(Representation):  # type: ignore
    __slots__ = ["__model_"]

    __model_: GensimFastText

    def __init__(
        self,
        train_algorithm: str = "cbow",
        min_n: int = 3,
        max_n: int = 6,
        bucket: int = 2_000_000,
        activation_method: str = "hierarchical_softmax",
        vector_size: int = 100,
        alpha: float = 0.025,
        min_alpha: float = 1e-4,
        window: int = 5,
        min_count: int = 5,
        negative: int = 5,
        epochs: int = 10,
        sorted_vocab: int = 1,
        workers: int = os.cpu_count() or 1,
        trained_model: Any | None = None,
        document_path: Path | str | None = None,
        train_corpus: list[str] | None = None,
        spacy_model_name: str = "en_core_web_sm",
        stop_word_removal_enabled: bool = True,
        stop_words: Optional[set[str]] = None,  # type: ignore
    ) -> None:
        super().__init__(stop_word_removal_enabled=stop_word_removal_enabled)
        self.features = np.arange(vector_size)
        self._representation = np.array([])

        if stop_word_removal_enabled and spacy_model_name.strip() != "":
            self._spacy_model = spacy.load(
                spacy_model_name, disable=["parser", "ner", "textcat", "taggers"]
            )

            self._stop_words = (
                stop_words if stop_words is not None and len(stop_words) > 0 else STOP_WORDS
            )  # type: ignore

        if trained_model:
            if isinstance(trained_model, GensimFastText):
                self.__model_ = trained_model
                return

            raise TypeError("model is not a valid type")

        assert train_corpus or document_path, "train_corpus or document_path must be provided"

        valid_train_algorithms = ["cbow", "skip-gram"]
        assert (
            train_algorithm.casefold().strip() in valid_train_algorithms
        ), f"model must be in {valid_train_algorithms}"

        valid_activation_methods = ["hierarchical_softmax", "negative_sampling"]
        assert (
            activation_method.casefold().strip() in valid_activation_methods
        ), f"activation_method must be in {valid_activation_methods}"

        self.__model_ = GensimFastText(
            sg=train_algorithm == 0 if train_algorithm.casefold().strip() == "cbow" else 1,
            min_n=min_n,
            max_n=max_n,
            hs=1 if activation_method.casefold().strip() == "hierarchical_softmax" else 0,
            bucket=bucket,
            vector_size=vector_size,
            alpha=alpha,
            window=window,
            negative=negative,
            min_alpha=min_alpha,
            min_count=min_count,
            epochs=epochs,
            sorted_vocab=sorted_vocab,
            workers=workers,
        )

        self.__train_model_(document_path=document_path, train_corpus=train_corpus)

    def __train_model_(
        self, document_path: Path | str | None, train_corpus: list[str] | None
    ) -> None:
        corpus_iterable = []

        if train_corpus:
            corpus_iterable = train_corpus
        else:
            corpus_iterable = list(FileIter(document_path))  # type: ignore

        corpus_iterable = [sentence.split() for sentence in self.pre_process(corpus_iterable)]

        self.__model_.build_vocab(
            corpus_iterable=corpus_iterable,
        )

        self.__model_.train(
            corpus_iterable=corpus_iterable,
            total_examples=self.__model_.corpus_count,
            epochs=self.__model_.epochs,
        )

    @classmethod
    def from_file(cls, model_filename: str | Path, stop_word_removal_enabled=True) -> "FastText":
        """
        Abstract Constructor to Create model from file
        """
        if os.path.exists(model_filename):
            return cls(
                trained_model=GensimFastText.load(model_filename, mmap="r"),
                stop_word_removal_enabled=stop_word_removal_enabled,
            )
        raise FileNotFoundError(f"model file '{model_filename}' not found")

    @classmethod
    def from_facebook_model(
        cls, model_path: str | Path, stop_word_removal_enabled=True
    ) -> "FastText":
        """
        Abstract Constructor to Create model from file
        """
        if os.path.exists(model_path):
            return cls(
                trained_model=load_facebook_model(model_path),
                stop_word_removal_enabled=stop_word_removal_enabled,
            )
        raise FileNotFoundError(f"model file '{model_path}' not found")

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


Representation.register(FastText)


if __name__ == "__main__":
    # TRAIN_CORPUS = [
    #     "Use this metaclass to create an ABC",
    #     "An ABC can be subclassed directly, and then acts as a mix-in class.",
    #     "You can also register unrelated concrete classes",
    #     "(even built-in  classes) and unrelated ABCs as ",
    #     "virtual subclasses” – these and their descendants will be considered",
    #     "subclasses of the registering ABC by the built"
    # ]

    with open(
        os.path.join(
            Path(__file__).parents[2],
            "shared",
            "datasets",
            "Reviews",
            "Sentiment Labelled",
            "imdb_labelled.train",
        ),
        "r+",
        encoding="utf-8",
    ) as file:
        TRAIN_CORPUS = [line.strip("\n") for line in file] * 5

        print(len(TRAIN_CORPUS))

    ft = FastText(
        train_corpus=TRAIN_CORPUS,
        min_count=1,
        vector_size=300,
        stop_word_removal_enabled=True,
        train_algorithm="skip-gram",
    )
    df = ft.generate_representation([*TRAIN_CORPUS], as_dataframe=True)
    print(df)
