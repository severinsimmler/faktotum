import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings, Sentence

Dataset = Dict[str, Dict[str, List[str]]]
Documents = Dict[str, List[str]]

random.seed(23)


def calculate_sentence_similarities(
    corpus: Union[Path, str],
    model: Union[Path, str],
    reference_sentences: List[str],
    n: int = 5000,
):
    logging.info("Constructing embedding...")
    embedding = Embedding(model)
    logging.info("Loading corpus and reference sentences...")
    with Path(corpus).open("r", encoding="utf-8") as file_:
        dataset = json.load(file_)
    with Path(reference_sentences).open("r", encoding="utf-8") as file_:
        ref = json.load(file_)
    logging.info("Sampling documents...")
    batch = select_documents(dataset, n)
    batch = dict(embedding.process_batch(batch))
    ref = dict(select_reference_sentences(dataset, ref))
    ref = dict(embedding.process_reference_sentences(ref))
    batch.update(ref)
    features = pd.DataFrame(batch).T
    features.to_csv("features.csv")
    similarities = sklearn.metrics.pairwise.cosine_similarity(features)
    return ref, pd.DataFrame(similarities, index=features.index, columns=features.index)


def select_new_sentences(matrix, ref):
    for name in ref:
        yield random.choice(matrix[name].sort_values()[:50].index)


def select_documents(dataset: Dataset, n: int = 5000):
    dataset = {name: data for name, data in dataset.items() if "en" not in name}
    selection = random.sample(list(dataset), n)
    return {name: data for name, data in dataset.items() if name in selection}


def select_reference_sentences(
    dataset: Dataset, reference_sentences: Dict[str, List[str]]
):
    i = 0
    for name, document in dataset.items():
        if name in reference_sentences:
            for index, sentence in document.items():
                if str(index) in reference_sentences[name]:
                    yield f"REFERENCE-{i}", sentence
                    i += 1


class Embedding:
    def __init__(self, filepath: Union[Path, str]):
        self._bert = BertEmbeddings(filepath)
        self.document_embeddings = DocumentPoolEmbeddings([self._bert])

    def get_vector(self, text: str) -> np.ndarray:
        sentence = Sentence(text, use_tokenizer=False)
        self.document_embeddings.embed(sentence)
        with torch.no_grad():
            vector = sentence.get_embedding()
            return vector.numpy()

    def process_reference_sentences(self, sentences: Dict[str, List[str]]):
        for name, sentence in sentences.items():
            try:
                sentence = " ".join(sentence)
                yield name, self.get_vector(sentence)
            except RuntimeError:
                logging.error("Oops! Reference sentence too long...")

    def process_batch(self, documents: Documents):
        for name, sentence in self.build_sentences(documents):
            logging.info(f"Processing {name}...")
            try:
                yield name, self.get_vector(sentence)
            except RuntimeError:
                logging.error("Oops! Sentence too long..")

    def build_sentences(self, documents: Documents):
        for name, sentences in documents.items():
            for index, sentence in sentences.items():
                yield f"{name}-{index}", " ".join(sentence)
