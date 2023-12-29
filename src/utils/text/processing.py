# (\s*|[^\s]*)([\"]+[^(\w|\d)])(\s*|[^\s]*)

"""
    Module containg all text processing functions
"""
import re
from unicodedata import normalize

SINGLE_QUOTES = "‘’'´`"
DOUBLE_QUOTES = '"”“'


def remove_additional_quotations(sentence: str, max_starting_word_size=3) -> str:
    """
        Remove aspas no texto que nao configuram abreviacoes
    """

    final_sentence = re.sub(
        fr"([{SINGLE_QUOTES}{DOUBLE_QUOTES}]+)([^\s{SINGLE_QUOTES}{DOUBLE_QUOTES}]+)([{SINGLE_QUOTES}{DOUBLE_QUOTES}]+)",
        r"\2", sentence
    )

    final_sentence = re.sub(
        fr"([{SINGLE_QUOTES}{DOUBLE_QUOTES}]+)(\w{{{str(max_starting_word_size)},}})",
        r" \2 ",
        final_sentence
    )

    final_sentence = re.sub(
        fr"(\w+)([{SINGLE_QUOTES}{DOUBLE_QUOTES}]+)([^\w\d])",
        r" \1 ",
        final_sentence
    )

    final_sentence = re.sub(
        fr"(\s*)([{SINGLE_QUOTES}{DOUBLE_QUOTES}])(\s*)([^\w])",
        r" ",
        final_sentence
    )

    return remove_additional_spaces(final_sentence)


def remove_punctuations(sentence: str) -> str:
    """
        Remove pontuacoes (exceto aspas) no texto
    """
    assert isinstance(sentence, str)

    return remove_additional_spaces(
        re.sub(r"[!#$%&\\()*+,\-./:;<=>?@[\\\]^_{|}~]", " ", sentence)
    )


def remove_accents(sentence: str) -> str:
    """
        Remove os acentos do texto
    """
    assert isinstance(sentence, str)

    return normalize('NFKD', sentence).encode('ASCII', 'ignore').decode('ASCII')


def remove_special_chars(sentence: str) -> str:
    """
        Remove caracteres como \n \b ou tags de HTML|XML no texto
    """
    assert isinstance(sentence, str)

    return remove_additional_spaces(
        re.sub(
            r"(\\+[A-RT-Za-rt-z]?|<\s*\/?\s*(\w|\d)*\s*\/\s*>)", " ", sentence)
    )


def remove_numbers(sentence: str) -> str:
    """
        Remove numeros no texto
    """
    assert isinstance(sentence, str)

    return remove_additional_spaces(
        re.sub(r"\d+(st|nd|rd|th)?", "", sentence, flags=re.IGNORECASE)
    )


def remove_additional_spaces(sentence: str) -> str:
    """
        Remove excesso de espacos no texto
    """
    assert isinstance(sentence, str)

    return re.sub(r"\s{2,}", r" ", sentence)


def remove_start_end_space(sentence: str) -> str:
    """
        Remove espacos no inicio e fim no texto
    """
    return re.sub(r"^\s+|\s+$", "", sentence)


def remove_apostrophe(sentence: str) -> str:
    """
        Remove todas aspas no texto
    """
    assert isinstance(sentence, str)

    return remove_additional_spaces(
        re.sub(fr"([{SINGLE_QUOTES}{DOUBLE_QUOTES}])", " ", sentence)
    )


def remove_all_spaces(sentence: str) -> str:
    """
        Remove todos espacos no texto
    """
    assert isinstance(sentence, str)

    return re.sub(r"\s", "", sentence)


def add_space_between_abbr_verbs(sentence: str) -> str:
    """
        Adiciona espaços entre as palavras abreviadas com aspas
    """
    assert isinstance(sentence, str)

    return re.sub(fr"(\w+)([{SINGLE_QUOTES}{DOUBLE_QUOTES}]+)(\w{{1,3}})", r"\1 '\3", sentence)


def sub_double_quotes_with_single_quotes(sentence: str) -> str:
    """
        Substitui todas aspas duplas por aspas simples
    """
    assert isinstance(sentence, str)

    return re.sub(f"[{DOUBLE_QUOTES}]", "'", sentence)

# def pre_process(stop_words: Set[str])\
#         -> Callable:
#     """
#         Retorna o texto após a remoção dos espaços
#         e qualquer caracter que não seja uma letra
#     """
#     def func(sentence: str) -> str:

#         assert isinstance(sentence, str)

#         assert isinstance(stop_words, Set)

#         # remove caracteres que não sejam letras
#         fixed_text: str = remove_accents(sentence)
#         # fixed_text = re.sub(r"[^A-Za-z\s'‘']", "", fixed_text)
#         # remove excesso de espaços
#         fixed_text = re.sub(r"\s{2,}", r" ", fixed_text).casefold()
#         # remove espacos no inicio e no final
#         fixed_text = re.sub(r"^\s+|\s+$", "", fixed_text)

#         if len(stop_words) > 0:
#             en_model: Any = spacy.load(
#                 "en_core_web_sm", disable=["parser", "ner"])

#             words: List[object] = en_model(fixed_text)

#             # remove os stop words
#             fixed_text = " ".join([
#                 token.lemma_.casefold()  # type: ignore
#                 for token in words
#                 if token.lemma_.casefold() not in stop_words  # type: ignore
#             ])

#         return re.sub(r"\s{2,}", " ", )
#     return func
