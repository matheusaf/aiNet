"""
    DBOW using Stanford Postagger (Stanford POS Tagger)
"""

from collections.abc import Generator
from typing import Any

import numpy as np
import spacy
import spacy_stanza
import stanza
from numba import from_dtype, jit, prange
from numba.core import types
from numba.typed import Dict  # pylint: disable = no-name-in-module
from pandas import DataFrame
from spacy import parts_of_speech, util  # type: ignore
from spacy.lang.en import English

from .representation import Representation

INT_TYPE = types.int64
UNICHAR_TYPE = from_dtype(np.dtype("<U5"))


# forces the function to compile nopython mode if not possible compilation will raise error
@jit(
    nopython=True,
    cache=True,
)
def generate_representation_helper(tagged_sentence: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    :return: A numpy array containing the frequency for each tag on the sentence.
    """

    tag_counter: Dict = Dict.empty(  # type: ignore
        key_type=UNICHAR_TYPE, value_type=INT_TYPE
    )

    i: int = 0
    feature_size: int = features.shape[0]
    representation_row = np.zeros(feature_size, dtype=np.float32)

    tags_size: int = tagged_sentence.shape[0]
    tag: str
    col_counter: int
    feature: str

    for i in range(tags_size):
        tag = tagged_sentence[i]

        if tag not in tag_counter:
            tag_counter[tag] = 0
        tag_counter[tag] += 1

    i = 0
    for i in range(feature_size):  # type: ignore
        feature = features[i]
        col_counter = tag_counter.get(feature)

        if col_counter is not None:
            representation_row[i] = col_counter

    return representation_row / tags_size


class STagger(Representation):
    """
    Stanford POS-Tagger
    """

    _spacy_model: English | Any

    def __init__(self, spacy_model_name: str = "en_core_web_sm") -> None:
        super().__init__(stop_word_removal_enabled=False)
        try:
            assert spacy_model_name in util.get_installed_models()
        except AssertionError:
            stanza.download("en")
        finally:
            self._spacy_model = spacy_stanza.load_pipeline(
                "en", use_gpu=True, processors="tokenize, pos", download_method=None
            )
            self.features = np.array(
                sorted(parts_of_speech.IDS.keys())[1:],  # type: ignore
                dtype="<U5",
            )
            self._representation = np.array([])

    def run_tagging_pipe(self, sentences: list[str] | str) -> Generator:
        """
        :param sentences: A list of sentences to be tagged.
        :return: A numpy array of the tagged sentences.
        """
        assert isinstance(sentences, list) or isinstance(
            sentences, str
        ), "The sentences must be a list of strings or a string."

        tagging_sentences = sentences

        if isinstance(sentences, str):
            tagging_sentences = [sentences]

        processed_sentences = list(self.pre_process(tagging_sentences))  # type: ignore

        # n_processes = (os.cpu_count() or 1) if len(processed_sentences) > 500 else 1

        docs = self._spacy_model.pipe(
            processed_sentences,
            disable=["lemmatizer", "parser", "ner", "textcat"],
            n_process=1,
        )

        for doc in docs:
            yield np.array([token.pos_ for token in doc], dtype="<U5")

    @jit(
        forceobj=True,
        parallel=True,
        nogil=True,
        # tries to release the global interpreter lock inside the compiled function.
        # The GIL will only be released if Numba can compile the function in nopython mode,
        # otherwise a compilation warning will be printed.
    )
    def generate_representation(
        self, sentences: list[str], as_dataframe: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        """
        :return: A tuple of the tags and their frequency.
        """
        spacy.prefer_gpu()  # type: ignore

        tagged_sentences: list[np.ndarray] = list(self.run_tagging_pipe(sentences))

        sentence_count: int = len(tagged_sentences)

        self._representation = np.zeros((sentence_count, self._features.shape[0]), np.float32)

        tagged_sentence: np.ndarray
        sentence_index: int = 0

        for sentence_index in prange(sentence_count):  # pylint: disable = not-an-iterable
            tagged_sentence = tagged_sentences[sentence_index]

            self._representation[sentence_index] = generate_representation_helper(
                tagged_sentence, self._features
            )

        if as_dataframe:
            return DataFrame(data=self._representation, columns=self._features)

        return (self.features, self.representation)


Representation.register(STagger)

if __name__ == "__main__":
    s = STagger()

    data = [
        "Use this metaclass to create an ABC.",
        "An ABC can be subclassed directly, and then acts as a mix-in class.",
        """You can also register unrelated concrete classes(even built-in classes)
                and unrelated ABCs as “virtual subclasses” – these and their descendants will be
                considered subclasses of the registering ABC by the built""",
    ] * 10

    df = s.generate_representation(
        data,
        as_dataframe=True,
    )

    print(df)
    print(df.shape)  # type: ignore
