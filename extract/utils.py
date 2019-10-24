"""
extract.utils
~~~~~~~~~~~~~

This module provides general helper functions.
"""

from typing import Generator

import syntok.segmenter
import syntok.tokenizer


TOKENIZER = syntok.tokenizer.Tokenizer()


def tokenize(text: str) -> Generator[str, None, None]:
    """Split text into tokens.

    Parameters
    ----------
    text
        The text to split into tokens.

    Yields
    ------
    One token at a time.
    """
    for token in TOKENIZER.tokenize(text):
        yield token.value.strip()


def sentencize(text: str, tokenize: bool = False) -> Generator[str, None, None]:
    """Split text into sentences.

    Parameters
    ----------
    text
        The text to split into tokens.
    tokenize
        If True, return tokenized sentences. Defaults to False.

    Yields
    ------
    One sentence at a time.
    """
    for paragraph in syntok.segmenter.process(text):
        for sentence in paragraph:
            if tokenize:
                yield [token.value.strip() for token in sentence]
            else:
                yield TOKENIZER.to_text(sentence).strip()
