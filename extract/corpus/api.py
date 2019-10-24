from pathlib import Path

from extract import utils


class Corpus:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()

    def documents(self):
        for document in self.directory.glob("*.txt"):
            yield document.read_text(encoding="utf-8")

    def tokens(self, **kwargs):
        for document in self.documents():
            yield utils.tokenize(document, **kwargs)

    def sentences(self):
        for document in self.documents():
            yield utils.sentencize(document)
