"""
faktotum.corpus.core
~~~~~~~~~~~~~~~~~~~

This module implements basic classes to process text corpora.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

from syntok.tokenizer import Token as _Token

from faktotum import utils


@dataclass
class Sentence:
    _tokens: Iterable[_Token]

    @property
    def tokens(self) -> Generator[str, None, None]:
        for token in self._tokens:
            yield str(token).strip()

    @property
    def text(self) -> str:
        return utils.TOKENIZER.to_text(self._tokens).strip()

    def __repr__(self):
        return f"<Sentence {hex(id(self))}>"


@dataclass
class Document:
    filepath: Path

    @property
    def name(self) -> str:
        return self.filepath.name

    @property
    def content(self) -> str:
        return self.filepath.read_text(encoding="utf-8")

    @property
    def sentences(self) -> Generator[Sentence, None, None]:
        for sentence in utils.sentencize(self.content):
            yield Sentence(sentence)

    @property
    def tokens(self) -> Generator[str, None, None]:
        for token in utils.tokenize(self.content):
            yield str(token).strip()

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence

    def __repr__(self):
        return f"<Document: {self.name}>"


@dataclass
class Corpus:
    documents: Iterable[Document]

    def __repr__(self):
        return f"<Corpus {hex(id(self))}>"

    def __iter__(self):
        for document in self.documents:
            yield document


@dataclass
class Token:
    text: str
    index: int
    label: str = None

    def __repr__(self):
        if self.label:
            return f"<Token {self.index}: {self.text} ({self.label})>"
        else:
            return f"<Token {self.index}: {self.text}>"
