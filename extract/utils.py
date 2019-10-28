"""
extract.utils
~~~~~~~~~~~~~

This module provides general helper functions.
"""

import logging
from typing import Generator

import syntok.segmenter
import syntok.tokenizer


TOKENIZER = syntok.tokenizer.Tokenizer()


def logger(name: str):
    """Generic tokenizer.

    Parameters
    ----------
    name
        Name of the logger.

    Returns
    -------
    A logger with log level INFO.
    """
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    return log


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
        text = token.value.strip()
        if text:
            yield text


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
                yield [token.value.strip() for token in sentence if token.value.strip()]
            else:
                yield TOKENIZER.to_text(sentence).strip()
