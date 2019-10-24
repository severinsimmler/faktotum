from typing import Generator

import regex as re
from spacy.lang.de import German


LANGUAGE = German()
LANGUAGE.max_length = 100000000
TOKEN_PATTERN = re.compile(r"\p{L}+\p{P}?\p{L}+")


def tokenize(
    text: str, lowercase: bool = False, only_words: bool = False
) -> Generator[str, None, None]:
    """Split text into tokens.

    Parameters
    ----------
    text
        The text to split into tokens.
    lowercase
        If True, all tokens will be lowercase. Defaults to False.
    only_words
        If True, numbers and punctuation will be ignored. Defaults to False.

    Yields
    ------
    One token at a time.
    """
    if only_words:
        for token in TOKEN_PATTERN.finditer(text):
            if lowercase:
                yield token.group(0).lower()
            else:
                yield token.group(0)
    else:
        for token in LANGUAGE(text):
            if lowercase:
                yield token.text.lower()
            else:
                yield token.text


def sentencize(text: str) -> Generator[str, None, None]:
    """Split text into sentences.

    Parameters
    ----------
    text
        The text to split into tokens.

    Yields
    ------
    One sentence at a time.
    """
    if "sentencizer" not in LANGUAGE.pipe_names:
        sentencizer = LANGUAGE.create_pipe("sentencizer")
        LANGUAGE.add_pipe(sentencizer)
    for sentence in LANGUAGE(text).sents:
        yield sentence
