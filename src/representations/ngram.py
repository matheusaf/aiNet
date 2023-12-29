"""
    Module containing Ngram implementation
"""

from typing import Optional, Union
from collections.abc import Iterator

import numpy as np
import spacy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

import utils.text.processing as text_processing

from .representation import Representation


class NGram(Representation):
    """
    Class representing NGram text representation
    """

    __slots__ = ["__min_n_", "__max_n_", "__model_type_"]

    __min_n_: int
    __max_n_: int
    __model_type_: str

    def __init__(
        self,
        min_ngram_group: int = 1,
        max_ngram_group: int = 1,
        stop_words: Optional[set[str]] = None,
        model_type: str = "tf-idf",
        spacy_model_name: str = "en_core_web_sm",
        vocabulary: np.ndarray | list[str] | None = None,
    ) -> None:
        assert isinstance(stop_words, (set, type(None))), "Stop words must be a set"

        assert (
            isinstance(min_ngram_group, int) and min_ngram_group > 0
        ), "min_ngram_group must be of type int and greater than 0"

        assert (
            isinstance(max_ngram_group, int) and min_ngram_group > 0
        ), "max_ngram_group must be of type int and greater than 0"

        assert max_ngram_group >= min_ngram_group, "max_ngram cannot be less than min_ngram"

        assert model_type.casefold().strip() in [
            "tf-idf",
            "count",
        ], "type must be ['count', 'tf-idf']"

        super().__init__(stop_word_removal_enabled=True)
        model_type = text_processing.remove_all_spaces(
            text_processing.remove_accents(model_type.casefold())
        )

        self.features = []
        self._representation = np.ndarray([])
        self.__model_type_ = model_type
        self.__min_n_ = min_ngram_group
        self.__max_n_ = max_ngram_group

        self._stop_words = stop_words if stop_words is not None else STOP_WORDS  # type: ignore

        self._spacy_model = spacy.load(
            spacy_model_name, disable=["parser", "ner", "textcat", "taggers"]
        )

        if vocabulary is not None:
            self.features = vocabulary

    @property
    def min_n(self) -> int:
        """
        Return the minimum ngram group
        """
        return self.__min_n_

    @property
    def max_n(self) -> int:
        """
        Return the maximum ngram group
        """
        return self.__max_n_

    def pre_process(self, sentences: list[str]) -> Iterator[str]:
        """
        function responsible for removing additional spaces and special characters
        : param sentence: sentence to be pre-processed
        : return: sentence pre-processed
        """
        processed_sentences = list(super().pre_process(sentences))

        processed_sentences = self.remove_stop_words(processed_sentences)

        for processed_sentence in processed_sentences:
            yield processed_sentence

    def _pre_process_sentence(self, sentence: str) -> str:
        processed_sentence = super()._pre_process_sentence(sentence)

        processed_sentence = text_processing.remove_punctuations(processed_sentence)
        processed_sentence = text_processing.remove_numbers(processed_sentence)
        processed_sentence = text_processing.add_space_between_abbr_verbs(processed_sentence)
        
        return text_processing.sub_double_quotes_with_single_quotes(processed_sentence)

    def __generate_representation_helper_(self, sentences: list[str]) -> np.ndarray:
        vectorizer: Union[CountVectorizer, TfidfVectorizer]
        vocabulary = self._features if self._features.shape[0] > 0 else None

        processed_sentences = list(self.pre_process(sentences))

        if self.__model_type_ == "count":
            vectorizer = CountVectorizer(
                lowercase=False,
                ngram_range=(self.__min_n_, self.__max_n_),
                vocabulary=vocabulary,
            )

        else:
            vectorizer = TfidfVectorizer(
                use_idf=True,
                lowercase=False,
                ngram_range=(self.__min_n_, self.__max_n_),
                vocabulary=vocabulary,
            )
        vectorizer_result = vectorizer.fit_transform(processed_sentences)\
            .toarray().astype(np.float32)  # type: ignore

        self.features = list(vectorizer.vocabulary_)

        return vectorizer_result

    def generate_representation(
        self, sentences: list[str], as_dataframe: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        self._representation = self.__generate_representation_helper_(sentences=sentences)

        if as_dataframe:
            return DataFrame(data=self._representation, columns=self._features)

        return (self.features, self.representation)


Representation.register(NGram)

if __name__ == "__main__":
    for n in (1, 3):
        for vectorizer_type in ["tf-idf", "count"]:
            print(f"\nn = {n} | vectorizer = {vectorizer_type}")
            ngram = NGram(min_ngram_group=n, max_ngram_group=n, model_type=vectorizer_type)
            df = ngram.generate_representation(
                [
                    "Use this metaclass to create an ABC.",
                    "An ABC can be subclassed directly, and then acts as a mix-in class.",
                    """You can also register unrelated concrete classes
                    (even built-in  classes) and unrelated ABCs as """,
                    """virtual subclasses” – these and their descendants will be considered
                subclasses of the registering ABC by the built""",
                ],
                as_dataframe=True,
            )

            print(df)
