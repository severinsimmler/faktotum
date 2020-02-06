"""
faktotum.ontologia.rules
~~~~~~~~~~~~~~~~~~~~~~~

This module implements basic rules to filter text corpora.
"""

from typing import Dict, Generator, List, Tuple

PlainCorpus = Dict[str, List[List[str]]]


def filter_corpus(
    corpus: PlainCorpus, words: List[str]
) -> Generator[PlainCorpus, None, None]:
    """Filters sentences in a corpus.

    Note
    ----
    This function does not support a :class:`Corpus` object.

    Parameters
    ----------
    corpus
        The corpus to filter.
    words
        The words which must be in a sentence.

    Yields
    ------
    One document at a time.
    """
    for name, document in corpus.items():
        sentences = list(filter_document(document, words))
        if sentences:
            yield name, sentences


def filter_document(
    document: List[List[str]], words: List[str]
) -> Generator[List[str], None, None]:
    """Filters sentences in a document.

    Note
    ----
    This function does not support a :class:`Document` object.

    Parameters
    ----------
    document
        The document to filter.
    words
        The words which must be in a sentence.

    Yields
    ------
    One sentence at a time.
    """
    for sentence in document:
        if any(word.lower() in " ".join(sentence).lower() for word in words):
            yield sentence
