"""
    Generating Dbow using MRC
"""

import os
import pathlib
import pickle
from collections.abc import Callable, Generator

import numpy as np
from pandas import DataFrame

from utils.text.processing import remove_all_spaces, remove_punctuations

from .representation import Representation
from .stagger import STagger


class MRC2(Representation):
    """
    The following methods are used to access the lexicon and category names.
    """

    __slots__ = ["_mrc_dic", "_stagger", "_dic_filepath", "_pickle_filepath"]

    _mrc_dic: dict
    _stagger: STagger
    _dic_filepath: str | pathlib.Path | None
    _pickle_filepath: str | pathlib.Path

    __numerical_ranges_ = [
        (0, 2),
        (2, 4),
        (4, 5),
        (5, 10),
        (10, 12),
        (12, 15),
        (15, 21),
        (21, 25),
        (25, 28),
        (28, 31),
        (31, 34),
        (34, 37),
        (37, 40),
        (40, 43),
    ]

    __numerical_category_names_ = [
        "NLET",
        "NPHON",
        "NSYL",
        "K-F-FREQ",
        "K-F-NCATS",
        "K-F-NSAMP",
        "T-L-FREQ",
        "BROWN-FREQ",
        "FAM",
        "CONC",
        "IMAG",
        "MEANC",
        "MEANP",
        "AOA",
        "DERIVATIONAL",
        "ABBREVIATION",
        "SUFFIX",
        "PREFIX",
        "HYPHENATED",
        "MULTI-WORD",
        "DIALECT",
        "ALIEN",
        "ARCHAIC",
        "COLLOQUIAL",
        "CAPITAL",
        "ERRONEOUS",
        "NONSENSE",
        "NONCE WORD",
        "OBSOLETE",
        "POETICAL",
        "RARE",
        "RHETORICAL",
        "SPECIALISED",
        "STANDARD",
        "SUBSTANDARD",
        "PRONUNCIATION_DIFFER_STRESS",
        "PRONUNCIATION_DIFFER",
        "CAPITALIZATION",
        "PLURAL",
        "SINGULAR",
        "BOTH_SINGULAR_PLURAL",
        "NO_PLURAL",
        "PLURAL_ACT_SINGULAR",
    ]

    """ _text_category_names = [
        "WTYPE",
        "PDWTYPE",
        "WORD",
        "PHON",
        "DPHON",
        "STRESS"
    ] """

    __universal_tag_to_wtype_ = {
        # "A": ("ADV"),
        # "C": ("CONJ", "CCONJ", "SCONJ"),
        # "I": ("INTJ"),
        # "J": ("ADJ", "NUM", "DET"),
        # "N": ("NOUN", "PROPN"),
        # "O": ("X", "PART"),
        # "P": ("VERB"),
        # "R": ("ADP"),
        # "U": ("PRON"),
        # "V": ("VERB", "AUX"),
        "PRON": ("U"),
        "VERB": ("V", "P"),
        "DET": ("J"),
        "NOUN": ("N"),
        "ADP": ("R"),
        "ADJ": ("J",),
        "CCONJ": ("C"),
        "AUX": ("V"),
        "ADV": ("A"),
        "PART": ("O"),
        "NUM": ("J"),
        "CONJ": ("C"),
        "INTJ": ("I"),
        "PROPN": ("N"),
        "SCONJ": ("C"),
        "X": ("O"),
    }

    """ _wtype_to_pdwtype_map = {
        "N": "NOUN",
        "V": "VERB",
        "J": "ADJ",
        "O": "X",
    } """

    def __init__(self, dic_filepath: str | pathlib.Path | None) -> None:
        super().__init__(stop_word_removal_enabled=False)
        self._pickle_filepath = os.path.dirname(__file__) + "/mrc.pickle"
        has_pickle: bool = os.path.exists(self._pickle_filepath)

        assert (isinstance(dic_filepath, (str, pathlib.Path)) and os.path.exists(dic_filepath)) or (
            has_pickle
        ), ValueError("dic_filepath is None")

        self.features = self.__numerical_category_names_
        self._representation = np.array([])
        self._stagger = STagger()
        self._mrc_dic = {}

        if has_pickle:
            self.__load_dict_()
        else:
            self._dic_filepath = dic_filepath
            self.__read_dict_()

    def __getstate__(self) -> dict:
        print(dir(self))
        print([*super().__slots__, *self.__slots__])
        obj_dict = {
            slot: getattr(self, slot)
            for slot in [*super().__slots__, *self.__slots__]
            if slot not in {"_stop_words", "_spacy_model", "_stagger"}
        }

        return dict(obj_dict)

    def __setstate__(self, state: dict) -> None:
        for slot, value in state.items():
            object.__setattr__(self, slot, value)

    def __handle_tq2_(self, value: str) -> int:
        """
        TQ2 Column -> the value Q means that word is a derivational variant of another.
        """
        return int(value.upper() == "Q")

    def __convert_number_(self, value: str) -> int:
        """
        Function to parse int values
        """
        try:
            return int(value, 10)
        except ValueError:
            return 0

    def __load_dict_(self) -> None:
        """
        Loads the dictionary from the pickle file.
        """
        with open(self._pickle_filepath, "rb") as file:
            self._mrc_dic = pickle.load(file)

    def __read_dict_(self) -> None:
        """
        Reads the dictionary file and stores it in a dictionary.
        """
        with open(self._dic_filepath, "r+", encoding="utf-8") as file:  # type: ignore
            self._mrc_dic = {
                word: value
                for line in file.readlines()
                for word, value in self.__process_dictionary_line_(line)
            }
            pickle.dump(self._mrc_dic, open(self._pickle_filepath, "wb"))

    def __process_dictionary_line_(
        self, line: str
    ) -> Generator[tuple[tuple[str, str], np.ndarray], None, None]:
        """
        Function to process each line of the dictionary
        """

        numerical_values: np.ndarray = np.zeros_like(
            self.__numerical_category_names_, dtype=np.int32
        )

        num_idx: int = 0
        # reading numerical properties
        for num_idx, (start_rng, end_rng) in enumerate(self.__numerical_ranges_):
            numerical_values[num_idx] = self.__convert_number_(line[start_rng:end_rng])

        # handling tq2 value as categorical
        num_idx += 1
        numerical_values[num_idx] = self.__handle_tq2_(line[43])

        # mapping the remaining categorical properties
        categorical_values: str = line[46:51]
        # function to map categorical values to columns
        map_index_category_functions: list[Callable] = [
            self.__map_alphsyl_index_,
            self.__map_status_index_,
            self.__map_var_index_,
            self.__map_cap_index_,
            self.__map_irreg_index_,
        ]

        # mapping categorical values
        for map_fn, value in zip(map_index_category_functions, categorical_values):
            if (col_idx := map_fn(value)) != -1:
                numerical_values[col_idx] += 1

        # handling text values
        text_values: list[str] = line[51:].split("|")
        word: str = text_values[0]

        # wtype
        wtype: str = remove_all_spaces(line[44].upper()) or str()

        # text_line_values = np.zeros_like(
        #     self._text_category_names, dtype=str
        # )\
        # .astype(object)
        # for value in text_values[1:]:
        #     text_line_values[txt_idx] = remove_all_spaces(value.upper())
        #     txt_idx += 1

        # np.r_['r', numerical_line_values, text_line_values]
        yield (word.lower(), wtype), numerical_values

    def __map_alphsyl_index_(self, value: str):
        """
        Converting ALPHSYL column to their respective index
        """
        final_value: str = value.upper()
        if final_value == "A":
            return self.__numerical_category_names_.index("ABBREVIATION")
        if final_value == "S":
            return self.__numerical_category_names_.index("SUFFIX")
        if final_value == "P":
            return self.__numerical_category_names_.index("PREFIX")
        if final_value == "H":
            return self.__numerical_category_names_.index("HYPHENATED")
        if final_value == "M":
            return self.__numerical_category_names_.index("MULTI-WORD")
        return -1

    def __map_status_index_(self, value: str) -> int:
        """
        Converting STATUS column to their respective index
        """
        final_value: str = value.upper()
        if final_value == "D":
            return self.__numerical_category_names_.index("DIALECT")
        if final_value == "F":
            return self.__numerical_category_names_.index("ALIEN")
        if final_value == "A":
            return self.__numerical_category_names_.index("ARCHAIC")
        if final_value == "Q":
            return self.__numerical_category_names_.index("COLLOQUIAL")
        if final_value == "C":
            return self.__numerical_category_names_.index("CAPITAL")
        if final_value == "N":
            return self.__numerical_category_names_.index("ERRONEOUS")
        if final_value == "E":
            return self.__numerical_category_names_.index("NONSENSE")
        if final_value == "W":
            return self.__numerical_category_names_.index("NONCE WORD")
        if final_value == "O":
            return self.__numerical_category_names_.index("OBSOLETE")
        if final_value == "P":
            return self.__numerical_category_names_.index("POETICAL")
        if final_value == "R":
            return self.__numerical_category_names_.index("RARE")
        if final_value == "H":
            return self.__numerical_category_names_.index("RHETORICAL")
        if final_value == "$":
            return self.__numerical_category_names_.index("SPECIALISED")
        if final_value == "S":
            return self.__numerical_category_names_.index("STANDARD")
        if final_value == "Z":
            return self.__numerical_category_names_.index("SUBSTANDARD")
        return -1

    def __map_var_index_(self, value: str) -> int:
        """
        Converting VAR column to their respective index
        """
        final_value: str = value.upper()
        if final_value == "O":
            return self.__numerical_category_names_.index("PRONUNCIATION_DIFFER_STRESS")
        if final_value == "B":
            return self.__numerical_category_names_.index("PRONUNCIATION_DIFFER")
        return -1

    def __map_cap_index_(self, value: str) -> int:
        """
        Converting CAP column to their respective index
        """
        final_value: str = value.upper()
        if final_value == "C":
            return self.__numerical_category_names_.index("CAPITALIZATION")
        return -1

    def __map_irreg_index_(self, value: str) -> int:
        """
        Converting IRREG column to their respective index
        """
        final_value: str = value.upper()
        if final_value == "Z":
            return self.__numerical_category_names_.index("PLURAL")
        if final_value == "Y":
            return self.__numerical_category_names_.index("SINGULAR")
        if final_value == "B":
            return self.__numerical_category_names_.index("BOTH_SINGULAR_PLURAL")
        if final_value == "N":
            return self.__numerical_category_names_.index("NO_PLURAL")
        if final_value == "P":
            return self.__numerical_category_names_.index("PLURAL_ACT_SINGULAR")
        return -1

    def _pre_process_sentence(self, sentence: str) -> str:
        processed_sentence = super()._pre_process_sentence(sentence)
        return remove_punctuations(processed_sentence)

    def __generate_representation_helper_(self, sentence: tuple[str, np.ndarray]) -> np.ndarray:
        """
        : return a dbow vector for a sentence
        """
        processed_sentence, tagged_sentence = sentence
        tokens = processed_sentence.split()

        row: np.ndarray = np.zeros(len(self._features), dtype=np.int32)

        mrc_value: np.ndarray | None

        for token, pos_tag in zip(tokens, tagged_sentence):
            wtypes: tuple[str] = self.__universal_tag_to_wtype_.get(pos_tag, ())
            for wtype in wtypes:
                if (mrc_value := self._mrc_dic.get((token, wtype), None)) is not None:
                    row += mrc_value
        return row / max(1, len(tokens))

    def generate_representation(
        self, sentences: list[str], as_dataframe: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | DataFrame:
        """
        : return a dbow matrix representing the sentences
        """
        self._representation = np.zeros((len(sentences), len(self._features)), dtype=np.float32)

        processed_sentences = list(self.pre_process(sentences))

        tagged_sentences = list(self._stagger.run_tagging_pipe(processed_sentences))

        for idx, processed_sentence in enumerate(processed_sentences):
            self._representation[idx] = self.__generate_representation_helper_(
                (processed_sentence, tagged_sentences[idx])
            )

        if as_dataframe:
            return DataFrame(data=self._representation, columns=self._features)

        return (self.features, self.representation)


Representation.register(MRC2)

if __name__ == "__main__":
    import time

    dic_path: pathlib.Path = (
        pathlib.Path(__file__).parents[2] / "shared/dictionaries/mrc" / "mrc2.dct"
    )
    mrc = MRC2(dic_filepath=dic_path)
    start = time.time()

    data = [
        "Use this metaclass to create an ABC.",
        "An ABC can be subclassed directly, and then acts as a mix-in class.",
        """You can also register unrelated concrete classes(even built-in classes)
        and unrelated ABCs as “virtual subclasses” – these and their descendants will be
        considered subclasses of the registering ABC by the built""",
    ] * 850

    print(len(data))

    df = mrc.generate_representation(
        data,
        as_dataframe=True,
    )
    print(time.time() - start)

    print(df)
