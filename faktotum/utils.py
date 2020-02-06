"""
faktotum.utils
~~~~~~~~~~~~~

This module provides general helper functions.
"""

from typing import Generator

import syntok.segmenter
import syntok.tokenizer
from transformers import AutoTokenizer

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


def normalize_bert_dataset(dataset, model_name_or_path, max_len=128):
    subword_len_counter = 0
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with open(dataset, "r", encoding="utf-8") as file_:
        for line in file_:
            line = line.strip()
            if not line:
                yield line
                subword_len_counter = 0
                continue
            token = line.split("\t")[1]
            current_subwords_len = len(tokenizer.tokenize(token))
            if current_subwords_len == 0:
                continue
            if (subword_len_counter + current_subwords_len) > max_len:
                yield ""
                yield line
                subword_len_counter = 0
                continue
            subword_len_counter += current_subwords_len
            yield line
