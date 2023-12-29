import os
from typing import Literal, Optional
from collections.abc import Iterator

import numpy as np
import spacy
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from spacy.lang.en.stop_words import STOP_WORDS

from utils.text.processing import remove_punctuations

from .representation import Representation

OutputType = Literal[None, "sentence_embedding", "token_embeddings"]


class SBert(Representation):
    __slots__ = [
        "__model_",
        "__model_name_",
        "__cache_folder_path_",
        "__batch_size_",
        "__show_progress_bar_",
        "__output_value_",
        # "__convert_to_numpy_",
        # "__convert_to_tensor_",
        "__normalize_embeddings_",
    ]

    __model_: SentenceTransformer
    __model_name_: str
    __cache_folder_path_: str
    __batch_size_: int
    __show_progress_bar_: bool
    __output_value_: OutputType
    # __convert_to_numpy_: bool
    # __convert_to_tensor_: bool
    __normalize_embeddings_: bool

    def __init__(
        self,
        model_name: str,
        batch_size=32,
        device=None,
        show_progress_bar=True,
        output_value: OutputType = "sentence_embedding",
        # convert_to_numpy=True,
        # convert_to_tensor=False,
        normalize_embeddings=False,
        spacy_model_name: str = "en_core_web_sm",
        stop_word_removal_enabled: bool = True,
        stop_words: Optional[set[str]] = None,  # type: ignore
    ) -> None:
        super().__init__(stop_word_removal_enabled=stop_word_removal_enabled)
        self.features = []
        self._representation = np.array([])
        self.__cache_folder_path_ = os.path.join(os.path.dirname(__file__), "bert_models_cache")

        if not os.path.exists(self.__cache_folder_path_):
            os.mkdir(self.__cache_folder_path_)

        valid_output_values = [None, "sentence_embedding", "token_embeddings"]

        assert output_value in valid_output_values, f"output_value must be in {valid_output_values}"

        if stop_word_removal_enabled:
            self._spacy_model = spacy.load(
                spacy_model_name, disable=["parser", "ner", "textcat", "taggers"]
            )

            self._stop_words = (
                stop_words if stop_words is not None and len(stop_words) > 0 else STOP_WORDS
            )  # type: ignore

        self.__batch_size_ = batch_size
        self.__output_value_ = output_value
        # self.__convert_to_numpy_ = convert_to_numpy
        # self.__convert_to_tensor_ = convert_to_tensor
        self.__model_name_ = model_name
        self.__show_progress_bar_ = show_progress_bar
        self.__normalize_embeddings_ = normalize_embeddings
        self.__model_ = SentenceTransformer(
            self.__model_name_, device=device, cache_folder=self.__cache_folder_path_
        )

    def pre_process(self, sentences: list[str]) -> Iterator[str]:
        processed_sentences = list(super().pre_process(sentences))

        processed_sentences = self.remove_stop_words(processed_sentences)

        for processed_sentence in processed_sentences:
            yield processed_sentence

    def _pre_process_sentence(self, sentence: str) -> str:
        processed_sentence = super()._pre_process_sentence(sentence)
        return remove_punctuations(processed_sentence)

    def generate_representation(
        self,
        sentences: list[str],
        as_dataframe: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        processed_sentences = list(self.pre_process(sentences))

        self._representation = self.__model_.encode(
            processed_sentences,  # type: ignore
            batch_size=self.__batch_size_,
            output_value=str(self.__output_value_),  # type: ignore
            convert_to_numpy=True,
            convert_to_tensor=False,
            show_progress_bar=self.__show_progress_bar_,
            normalize_embeddings=self.__normalize_embeddings_,
        )

        self._representation = self._representation.astype(np.float32)  # type: ignore

        self._features = np.arange(
            self._representation.shape[1]  # type: ignore
        )

        if as_dataframe:
            return DataFrame(
                data=self._representation,  # type: ignore
                columns=self._features,
            )

        return (self.features, self.representation)


Representation.register(SBert)


if __name__ == "__main__":
    bert = SBert("stsb-roberta-large", stop_word_removal_enabled=True)

    df = bert.generate_representation(
        [
            "Use this metaclass to create an ABC.",
            "An ABC can be subclassed directly, and then acts as a mix-in class.",
            """You can also register unrelated concrete classes(even built-in classes)
        and unrelated ABCs as “virtual subclasses” – these and their descendants will be
        considered subclasses of the registering ABC by the built""",
        ],
        as_dataframe=True,
    )
    
    print(df)
