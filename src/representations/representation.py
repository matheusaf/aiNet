"""
    Base Analyzer abstract class.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Optional

import numpy as np
from pandas import DataFrame
from spacy.language import Language

import utils.text.processing as text_processing


class Representation(metaclass=ABCMeta):
    """
    Abstract class for all representations.
    """

    __slots__ = [
        "_representation",
        "_features",
        "_spacy_model",
        "_is_stop_word_removal_enabled",
        "_stop_words",
    ]

    _features: np.ndarray
    _spacy_model: Optional[Language]
    _representation: np.ndarray
    _is_stop_word_removal_enabled: bool
    _stop_words: Optional[set[str]]

    def __init__(self, stop_word_removal_enabled: bool = False) -> None:
        self._is_stop_word_removal_enabled = stop_word_removal_enabled

    @classmethod
    def __subclasshook__(cls, subclass: Representation) -> bool:
        """
        :param subclass: The subclass to be checked.
        """
        return hasattr(subclass, "generate_dbow") and callable(subclass.generate_representation)

    @property
    def stop_words(self) -> set[str] | None:
        """
        :return: A numpy array of all of the features represented.
        """

        if self._stop_words is not None:
            return {*self._stop_words}

        return self._stop_words

    @stop_words.setter
    def stop_words(self, value: Optional[set[str]]) -> None:
        """
        :return: A numpy array of all of the features represented.
        """

        self._stop_words = value

    @property
    def features(self) -> np.ndarray:
        """
        :return: A numpy array of all of the features represented.
        """

        return self._features.copy()

    @features.setter
    def features(self, new_features: np.ndarray | list[str]) -> None:
        self._representation = np.zeros((len(new_features), 1))

        if isinstance(new_features, list):
            self._features = np.array(new_features)
            return
        self._features = new_features

    @property
    def representation(self) -> np.ndarray:
        """
        :return: A dataframe of the tags and their frequency.
        """
        return self._representation.copy()

    @representation.setter
    def representation(self, value: np.ndarray) -> None:
        """
        :param value: A dataframe of the tags and their frequency.
        """
        assert isinstance(value, np.ndarray)

        self._representation = value

    @property
    def remove_stop_words_flag(self) -> bool:
        """
        :return: A boolean indicating if stop word removal is enabled.
        """
        return self._is_stop_word_removal_enabled

    @abstractmethod
    def generate_representation(
        self, sentences: list[str], as_dataframe: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        """
        Create and a dataframe of the tags and their frequency.
        :param sentences: A list of sentences to be tagged.
        :param as_dataframe: If the representation should be returned as a
            dataframe.
        :return: A dataframe of the tags and their frequency.
        """
        raise NotImplementedError()

    def pre_process(self, sentences: list[str]) -> Iterator[str]:
        """
        Pre process a list of sentences.
        :return: a list of processed sentences.
        """
        assert isinstance(sentences, list) and all(
            isinstance(sentence, str) for sentence in sentences
        )

        for sentence in sentences:
            yield self._pre_process_sentence(sentence)

    def _pre_process_sentence(self, sentence: str) -> str:
        """
        Pre process a single sentence.
        :return: The pre processed sentence.
        """
        assert isinstance(sentence, str)

        final_sentence = sentence.casefold()
        final_sentence = text_processing.remove_accents(final_sentence)
        final_sentence = text_processing.remove_special_chars(final_sentence)
        final_sentence = text_processing.remove_additional_spaces(final_sentence)
        final_sentence = text_processing.remove_start_end_space(final_sentence)

        return text_processing.remove_additional_quotations(final_sentence)

    def remove_stop_words(
        self, sentences: list[str]
    ) -> list[str] | np.ndarray:
        """
        Remove the stop words from the sentences.
        """
        assert isinstance(sentences, list) and all(
            isinstance(sentence, str) for sentence in sentences
        )

        if (
            not self._is_stop_word_removal_enabled
            or self._spacy_model is None
            or self._stop_words is None
        ):
            return sentences

        n_processes = 1

        # if not spacy.prefer_gpu(): # type: ignore
        #     n_processes = (os.cpu_count() or 1) if len(sentences) > 2_500 else 1

        docs = self._spacy_model.pipe(
            sentences,
            disable=["ner", "textcat", "parser"],
            n_process=n_processes,
        )

        return [
            " ".join(
                [
                    valid_token
                    for token in doc
                    if (valid_token := token.lemma_.casefold()) not in self._stop_words
                ]
            )
            for doc in docs
        ]
