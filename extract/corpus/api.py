"""
extract.corpus.api
~~~~~~~~~~~~~~~~~~

This module implements the high-level API to process text corpora.
"""

from pathlib import Path
from typing import Generator, List, Union

from extract import utils


log = utils.logger(__file__)


class Corpus:
    def __init__(self, directory: Union[Path, str]):
        self.directory = Path(directory).resolve()
        if not self.directory.exists():
            raise OSError(f"The directory {self.directory} does not exist.")

    def documents(self, yield_name: bool = False) -> Generator[str, None, None]:
        """Documents as plain strings."""
        try:
            for document in self.directory.glob("*.txt"):
                log.debug(f"Processing {document.stem}...")
                if yield_name:
                    yield document.name, document.read_text(encoding="utf-8")
                else:
                    yield document.read_text(encoding="utf-8")
        except StopIteration:
            raise StopIteration(
                f"The directory {self.directory} does not contain any .txt files."
            )

    def tokens(self, **kwargs) -> Generator[List[str], None, None]:
        """Documents as tokens."""
        for document in self.documents():
            yield list(utils.tokenize(document, **kwargs))

    def sentences(self, tokenize: bool = True) -> Generator[List[str], None, None]:
        """Documents as sentences."""
        for document in self.documents():
            for sentence in utils.sentencize(document, tokenize=tokenize):
                yield sentence
