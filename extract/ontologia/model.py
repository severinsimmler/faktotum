from pathlib import Path

import gensim


class Model:
    def __init__(self, filepath: Path):
        self.fasttext = gensim.models.fasttext.load_facebook_model(filepath)

    def train(self, corpus, epochs: int = 10):
        self.fasttext.build_vocab(corpus, update=True)
        total_examples = self.fasttext.corpus_count
        self.fasttext.train(
            sentences=corpus, total_examples=total_examples, epochs=epochs
        )

    def most_similar(self, token, n=10):
        return [token[0] for token in self.fasttext.wv.most_similar([token], topn=n)]
