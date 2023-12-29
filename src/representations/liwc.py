"""
# ====================================================== #
#   Adaptation of LIWC module to allow multiprocessing   #
# ------------------------------------------------------ #
#           original module source below:                #
#        https: // github.com/chbrown/liwc-python        #
# ====================================================== #
"""

import os
import pathlib
from collections import Counter
from collections.abc import Generator
from io import TextIOWrapper

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from utils.text.processing import remove_punctuations

from .representation import Representation


class LIWC(Representation):
    """
        The following methods are used to access the lexicon and category names.
    """

    __slots__ = [
        "__trie_",
        "__lexicon_"
    ]

    __trie_: dict
    __lexicon_: dict

    def __init__(self, dic_filepath: str | pathlib.PurePath) -> None:
        super().__init__(stop_word_removal_enabled=False)
        assert os.path.exists(dic_filepath), "LIWC dictionary file not found"
        self.features = []
        self._representation = np.ndarray([])
        self.__read_dic_(dic_filepath)
        self.__build_trie_()

    @property
    def lexicon(self) -> dict:
        """
            Return the lexicon mapping.
        """
        return {**self.__lexicon_}

    @property
    def trie(self) -> dict:
        """
            Return the character-trie.
        """
        return {**self.__trie_}

    # type: ignore
    def __read_dic_(self, filepath: str | pathlib.PurePath) -> None:
        """
            Reads a LIWC lexicon from a file in the .dic format,
                returning a tuple of (lexicon, category_names), where:
                    * `lexicon` is a dict mapping string patterns to lists of category names in
                    * `category_names` is a list of category names (as strings)
        """
        with open(file=filepath, mode="r", encoding="utf-8") as file:
            # read up to first "%" (should  be very first line of file)
            for line in file:
                if line.strip() == "%":
                    break
            # read categories (a mapping from integer string to category name)
            category_mapping = dict(self.__parse_categories_(file))
            # read lexicon (a mapping from matching string to a list of category names)
            self.__lexicon_ = dict(
                self.__parse_lexicon_(file, category_mapping))

            self.features = [
                "WC",
                *list(category_mapping.values())
            ]

    def __parse_categories_(self, lines: TextIOWrapper) -> Generator:
        """
            Read (category_id, category_name) pairs from the categories section.
            Each line consists of an integer followed a tab and then the category name.
            This section is separated from the lexicon by a line consisting of a single "%".
        """
        for line in lines:
            line = line.strip()
            if line == "%":
                return
            # ignore non-matching groups of categories
            if "\t" in line:
                category_id, category_name = line.split("\t", 1)
                yield category_id, category_name

    def __parse_lexicon_(self, lines: TextIOWrapper, category_mapping: dict) -> Generator:
        """
            Read (match_expression, category_names) pairs from the lexicon section.
            Each line consists of a match expression followed by a tab and then one or more one
            tab-separated integers, which are mapped to category names using `category_mapping`.
        """
        for line in lines:
            line = line.strip()
            parts = line.split("\t")
            yield parts[0], [category_mapping[category_id] for category_id in parts[1:]]

    def __build_trie_(self) -> None:
        """
            Build a character-trie from the plain pattern_string -> categories_list
            mapping provided by `lexicon`.
            Some LIWC patterns end with a `*` to indicate a wildcard match.
        """
        self.__trie_ = {}
        for pattern, category_names in self.__lexicon_.items():
            cursor = self.__trie_
            for char in pattern:
                if char == "*":
                    cursor["*"] = category_names
                    break
                if char not in cursor:
                    cursor[char] = {}
                cursor = cursor[char]
            cursor["$"] = category_names

    def __search_trie_(
        self,
        trie: dict,
        token: str,
        token_i: int = 0
    ) -> list[None] | list[str]:
        """
            Search the given character-trie for paths that match the `token` string.
        """
        if "*" in trie:
            return trie["*"]
        if "$" in trie and token_i == len(token):
            return trie["$"]
        if token_i < len(token):
            char = token[token_i]
            if char in trie:
                return self.__search_trie_(trie=trie[char], token=token, token_i=token_i + 1)
        return []

    def __parse_token_(self, token: str) -> Generator:
        """
            Parse a token into a list of category names.
        """
        for category_name in self.__search_trie_(trie=self.__trie_, token=token):
            yield category_name

    def __generate_representation_helper_(self, result: Counter, word_count: int) -> np.ndarray:
        """
            Metodo helper que gera cada linha do dbow formado para o metodo liwc
        """
        row = np.zeros(len(self._features), dtype=np.float32)

        if word_count > 0:
            for index, category in enumerate(self._features):
                if result.get(category) is not None:
                    row[index] += result.get(category)
            # a sentence will be represented by averaging each liwc word vector
            row = np.round(row / np.float32(word_count), 8)
            # store the word count as the first column
            row[0] = word_count

        return row

    def _pre_process_sentence(self, sentence: str) -> str:
        # pylint: disable=unused-private-member
        processed_sentence = super()._pre_process_sentence(sentence)
        return remove_punctuations(processed_sentence)

    def generate_representation(
        self,
        sentences: list[str],
        as_dataframe: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        # -> pd.DataFrame:
        """
        : return A dataframe of the tags and their frequencies
        """
        self._representation = np.zeros(
            (len(sentences), len(self._features)),
            dtype=np.float32
        )

        for idx, processed_sentence in enumerate(self.pre_process(sentences)): # type: ignore
            tokens = processed_sentence.split()
            result = Counter(
                [
                    category
                    for token in tokens
                    for category in self.__parse_token_(token)
                ]
            )
            current_word_count = len(tokens)
            self._representation[idx] = self.__generate_representation_helper_(
                result=result, word_count=current_word_count
            )

        # normalizing WORD COUNT attribute
        self._representation[:, [0]] = MinMaxScaler()\
            .fit_transform(self._representation[:, [0]])

        if as_dataframe:
            return DataFrame(
                data=self._representation,
                columns=self._features
            )

        return (self.features, self.representation)


Representation.register(LIWC)

if __name__ == "__main__":
    dic_path: pathlib.PurePath = pathlib.Path(
        __file__
    ).parents[2] / "shared" / "dictionaries" / "liwc" / "LIWC2015.dic"

    assert os.path.exists(dic_path)

    liwc = LIWC(dic_path)

    df = liwc.generate_representation([
        "Use this metaclass to create an ABC.",
        "An ABC can be subclassed directly, and then acts as a mix-in class.",
        """You can also register unrelated concrete classes
            (even built-in classes) and unrelated ABCs as """,
        '''"virtual subclasses” – these and their descendants will be considered
        subclasses of the registering ABC by the built'''
    ], as_dataframe=True)

    print(df)
