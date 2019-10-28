"""
extract.ontologia.api
~~~~~~~~~~~~~~~~~~~~~

This module implements the high-level API to define an ontology.
"""

from pathlib import Path
from typing import Iterable, List, Optional

import gensim

import extract


log = extract.logger(__file__)


class FastText:
    def __init__(self, filepath: Optional[Path] = None):
        if filepath:
            self.pretrained = True
            if filepath.suffix == ".bin":
                log.info("Loading pre-trained Facebook fastText model...")
                self.model = gensim.models.fasttext.load_facebook_model(filepath)
            else:
                log.info("Loading pre-trained custom fastText model...")
                self.model = gensim.models.fasttext.FastText.load(filepath)
        else:
            log.info("Constructing plain fastText model...")
            self.pretrained = False
            self.model = gensim.models.fasttext.FastText(
                size=300, window=5, sg=0, negative=10, min_n=5, max_n=5, seed=23
            )

    def train(self, corpus: Iterable[List[str]], epochs: int = 10):
        if self.pretrained:
            log.info("Updating vocabulary...")
            self.model.build_vocab(corpus, update=True)
        else:
            log.info("Building vocabulary...")
            self.modle.build_vocab(corpus)
        log.info("Start training...")
        self.model.train(
            sentences=corpus, total_examples=self.model.corpus_count, epochs=epochs
        )
        log.info("Training was successful!")

    def most_similar(self, token: str, n: int = 10) -> List[str]:
        return [token[0] for token in self.model.wv.most_similar([token], topn=n)]
