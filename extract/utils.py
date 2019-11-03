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
        yield str(token).strip()


def sentencize(text: str) -> Generator[str, None, None]:
    """Split text into sentences.

    Parameters
    ----------
    text
        The text to split into tokenized sentences.

    Yields
    ------
    One sentence at a time.
    """
    for paragraph in syntok.segmenter.process(text):
        for sentence in paragraph:
            yield sentence
