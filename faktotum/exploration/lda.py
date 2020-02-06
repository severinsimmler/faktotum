"""
faktotum.exploration.lda
~~~~~~~~~~~~~~~~~~~~~~~

This module implements classes to process MALLET-produced topic models.
"""

import collections
import json
from pathlib import Path

import pandas as pd


class TopicModel:
    def __init__(
        self,
        topics_filepath: Path,
        document_topics_filepath: Path,
        stopword_filepath: Path = None,
    ):
        self.topics_filepath = topics_filepath
        self.document_topics_filepath = document_topics_filepath
        self.stopword_filepath = stopword_filepath
        if self.stopword_filepath:
            self.stopwords = json.loads(
                self.stopword_filepath.read_text(encoding="utf-8")
            )
        else:
            self.stopwords = None
        self.topics = pd.DataFrame(
            list(self._read_topics()),
            columns=["dirichlet"] + [f"word{i}" for i in range(20)],
        )
        self.document_topics = pd.DataFrame(dict(self._read_document_topics()))

    def _read_topics(self):
        with self.topics_filepath.open("r", encoding="utf-8") as topics_file:
            for line in topics_file:
                dirichlet = line.split("\t")[1]
                sequence = line.split("\t")[2]
                if self.stopwords:
                    yield [dirichlet] + [
                        token.strip()
                        for token in sequence.split(" ")
                        if token not in self.stopwords and token.strip()
                    ][:20]
                else:
                    yield [dirichlet] + [
                        token.strip() for token in sequence.split(" ") if token.strip()
                    ][:20]

    def _read_document_topics(self):
        with self.document_topics_filepath.open(
            "r", encoding="utf-8"
        ) as document_topics_file:
            for line in document_topics_file:
                name = Path(line.split("\t")[1]).name
                yield name, [float(score) for score in line.split("\t")[2:]]

    def most_frequent_words(self, n: int = 500):
        mfw = collections.Counter()
        for _, topic in self.topics.iterrows():
            mfw.update(list(topic))
        return mfw.most_common(n)

    def dominant_topics(self, n: int = 1, return_name: bool = False):
        for document, values in self.document_topics.iteritems():
            if return_name:
                dominant = list(values.sort_values(ascending=False)[:n].index)
                if n == 1:
                    dominant = dominant[0]
                yield document, dominant
            else:
                dominant = list(values.sort_values(ascending=False)[:n].index)
                if n == 1:
                    dominant = dominant[0]
                yield dominant

    def count_dominant_topics(self):
        counter = collections.Counter()
        for dominant_topic in self.dominant_topics():
            counter.update([dominant_topic])
        return counter
