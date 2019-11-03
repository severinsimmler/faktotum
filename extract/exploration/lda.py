"""
extract.exploration.lda
~~~~~~~~~~~~~~~~~~~~~~~

This module implements classes to process MALLET-produced topic models.
"""

import collections
from dataclasses import dataclass

import numpy as np


@dataclass
class TopicModel:
    topics_filepath: Path
    document_topics_filepath: Path

    def _read_topics(self):
        with self.topics_filepath.open("r", encoding="utf-8") as topics_file:
            for line in topics_file:
                sequence = row.split("\t")[2]
                yield sequence.split(" ")[:250]

    def _read_document_topics(self):
        with self.document_topics_filepath.open(
            "r", encoding="utf-8"
        ) as document_topics_file:
            for line in document_topics_file:
                yield [float(score) for score in line.split("\t")[2:]]

    @property
    def topics(self):
        topics_ = list(self._read_topics())
        return np.matrix(topics_)

    @property
    def document_topics(self):
        document_topics_ = list(self._read_document_topics())
        return np.matrix(document_topics_)

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
