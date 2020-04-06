"""
faktotum.utils
~~~~~~~~~~~~~

This module provides general helper functions.
"""

from typing import Generator

import syntok.segmenter
import syntok.tokenizer
from transformers import AutoTokenizer
import numpy as np
import torch


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
            token = line.split(" ")[0]
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


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss
