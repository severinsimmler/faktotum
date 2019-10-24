"""
extract.ontologia.api
~~~~~~~~~~~~~~~~~~~~~

This module implements the high-level API to define an ontology.
"""

from pathlib import Path
from typing import Iterable, List

import gensim


class FastText:
    def __init__(self, filepath: Path):
        self.model = gensim.models.fasttext.load_facebook_model(filepath)

    def train(self, corpus: Iterable[List[str]], epochs: int = 10):
        self.model.build_vocab(corpus, update=True)
        self.model.train(
            sentences=corpus, total_examples=self.model.corpus_count, epochs=epochs
        )

    def most_similar(self, token: str, n: int = 10) -> List[str]:
        return [token[0] for token in self.model.wv.most_similar([token], topn=n)]
