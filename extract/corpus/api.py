"""
extract.corpus.api
~~~~~~~~~~~~~~~~~~

This module implements the high-level API for corpus processing.
"""

from pathlib import Path
from typing import Generator, List

from extract import utils

log = utils.logger(__file__)


class Corpus:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        if not self.directory.exists():
            raise OSError(f"The directory {self.directory} does not exist.")

    def documents(self) -> Generator[str, None, None]:
        try:
            for document in self.directory.glob("*.txt"):
                log.debug(f"Processing {document.stem}...")
                yield document.read_text(encoding="utf-8")
        except StopIteration:
            raise StopIteration(
                f"The directory {self.directory} does not contain any .txt files."
            )

    def tokens(self, **kwargs) -> Generator[List[str], None, None]:
        for document in self.documents():
            yield list(utils.tokenize(document, **kwargs))

    def sentences(self) -> Generator[List[str], None, None]:
        for document in self.documents():
            yield list(utils.sentencize(document))
