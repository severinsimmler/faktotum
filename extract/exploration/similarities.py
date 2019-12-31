import random
from pathlib import Path
from typing import Dict, List, Union

from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings, Sentence
import torch
import pandas as pd
import numpy as np
import sklearn.metrics


Dataset = Dict[str, Dict[str, List[str]]]
Documents = Dict[str, List[str]]

random.seed(23)


def calculate_document_similarities(
    corpus: Union[Path, str], model: Union[Path, str], n: int = 5000
):
    embedding = Embedding(model)
    with Path(corpus).open("r", encoding="utf-8") as file_:
        dataset = json.load(file_)
    batch = select_documents(dataset, n)
    features = pd.DataFrame(embedding.process_batch(batch)).T
    similarities = sklearn.metrics.pairwise.cosine_similarity(features)
    return pd.DataFrame(similarities, index=features.index, columns=features.index)


def select_documents(dataset: Dataset, n: int = 5000):
    selection = random.sample(dataset, n)
    return {name: data for name, data in dataset.items() if name in selection}


class Embedding:
    def __init__(self, filepath: Union[Path, str]):
        self._bert = BertEmbeddings(filepath)
        self.document_embeddings = DocumentPoolEmbeddings([self._bert])

    def get_vector(text: str) -> np.ndarray:
        sentence = Sentence(text, use_tokenizer=False)
        self.document_embeddings.embed(sentence)
        with torch.no_grad():
            vector = sentence.get_embedding()
            return vector.numpy()

    def process_batch(documents: Documents):
        for name, sentence in self.build_sentences(documents):
            yield name, self.get_vector(sentence)

    def build_sentences(documents: Documents):
        for name, sentences in documents.items():
            for index, sentence in sentences.items():
                yield f"{name}-{index}", " ".join(sentence)
