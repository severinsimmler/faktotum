"""
faktotum.utils
~~~~~~~~~~~~~

This module provides general helper functions.
"""

from typing import Generator

import syntok.segmenter
import syntok.tokenizer

TOKENIZER = syntok.tokenizer.Tokenizer()
MODELS = {
    "ner": {
        "literary-texts": "severinsimmler/literary-german-bert",
        "press-texts": "severinsimmler/german-press-bert",
    },
    "ned": {
        "literary-texts": "severinsimmler/literary-german-bert",
        "press-texts": "severinsimmler/bert-base-german-press-cased",
    },
}


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


def pool_entity(indices, features):
    entity = features[indices[0]]
    for index in indices[1:]:
        entity += features[index]
    return entity


def extract_features(pipeline: Pipeline, sentence: str) -> Entities:
    vectors = list()
    for token_id, vector in zip(
        pipeline.tokenizer.encode(sentence), np.squeeze(pipeline(sentence))
    ):
        token = pipeline.tokenizer.decode([token_id])
        if token not in {"[CLS]", "[SEP]", "[MASK]"} and not token.startswith("##"):
            vectors.append(vector)
    return vectors