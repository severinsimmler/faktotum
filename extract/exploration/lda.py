"""
extract.exploration.lda
~~~~~~~~~~~~~~~~~~~~~~~

This module implements classes to process MALLET-produced topic models.
"""

import collections
import json
from pathlib import Path

import numpy as np


class TopicModel:
    def __init__(self, topics_filepath: Path, document_topics_filepath: Path, stopword_filepath: Path):
        self.topics_filepath = topics_filepath
        self.document_topics_filepath = document_topics_filepath
        self.stopword_filepath = stopword_filepath
        self.stopwords = json.loads(self.stopword_filepath.read_text(encoding="utf-8"))
        self.topics = np.array(list(self._read_topics()))
        self.document_topics = np.array(list(self._read_document_topics()))

    def _read_topics(self):
        with self.topics_filepath.open("r", encoding="utf-8") as topics_file:
            for line in topics_file:
                sequence = line.split("\t")[2]
                yield [token.strip() for token in sequence.split(" ") if token not in self.stopwords and token.strip()][:20]

    def _read_document_topics(self):
        with self.document_topics_filepath.open(
            "r", encoding="utf-8"
        ) as document_topics_file:
            for line in document_topics_file:
                yield [float(score) for score in line.split("\t")[2:]]

    def most_frequent_words(self, n: int = 500):
        mfw = collections.Counter()
        for topic in self.topics:
            mfw.update(topic)
        return mfw.most_common(n)

    def dominant_topics(self):
        for document in self.document_topics:
            yield document.argmax()

    def count_dominant_topics(self):
        counter = collections.Counter()
        for dominant_topic in self.dominant_topics:
            counter.update([dominant_topic])
