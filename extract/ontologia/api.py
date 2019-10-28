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
    def __init__(self, pretrained_model: Optional[Path] = None):
        if pretrained_model:
            self.pretrained = True
            if pretrained_model.suffix == ".bin":
                log.info("Loading pre-trained Facebook fastText model...")
                self.model = gensim.models.fasttext.load_facebook_model(
                    str(pretrained_model)
                )
            else:
                log.info("Loading pre-trained custom fastText model...")
                self.model = gensim.models.fasttext.FastText.load(str(pretrained_model))
        else:
            log.info("Constructing plain fastText model...")
            self.pretrained = False
            self.model = gensim.models.fasttext.FastText(
                size=300, window=5, sg=0, negative=10, min_n=5, max_n=5, seed=23
            )

    def train(self, corpus: Iterable[List[str]], epochs: int = 10):
        """Train the model.

        Parameters
        ----------
        corpus
            The tokenized corpus.
        epochs
            The number of epochs to train.
        """
        if self.pretrained:
            log.info("Updating vocabulary...")
            self.model.build_vocab(corpus, update=True)
        else:
            log.info("Building vocabulary...")
            self.model.build_vocab(corpus)
        log.info("Start training...")
        self.model.train(
            sentences=corpus, total_examples=self.model.corpus_count, epochs=epochs
        )
        log.info("Training was successful!")

    def most_similar(self, token: str, n: int = 10) -> List[str]:
        """Get the most similar words.

        Paramters
        ---------
        token
            The token to get similar words to.
        n
            The number of most similar words.
        """
        return [token[0] for token in self.model.wv.most_similar([token], topn=n)]
