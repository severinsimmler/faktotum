import json
import logging
from pathlib import Path

import pandas as pd

from faktotum.clustering.utils import Clustering, Embeddings

logging.basicConfig(format="%(message)s", level=logging.INFO)


def load_data():
    package_folder = Path(__file__).parent.parent
    data_folder = Path(package_folder, "data", "droc", "linking")
    data = dict()
    for file_ in data_folder.glob("*.txt"):
        with file_.open("r", encoding="utf-8") as file_:
            data.update(json.load(file_))
    return data


def compare_embeddings(model_directory):
    stats = list()
    index = list()

    data = load_data()
    embeddings = Embeddings(model_directory, "gutenberg")

    for embedding, model in [
        ("CBOW_{w2v}", embeddings.cbow_w2v),
        ("Skipgram_{w2v}", embeddings.skipgram_w2v),
        ("Facebook CBOW_{ft}", embeddings.cbow_ft_fb),
        ("CBOW_{ft}", embeddings.cbow_ft),
        ("Skipgram_{ft}", embeddings.skipgram_ft),
        ("dBERT", embeddings.bert_g),
        ("dBERT_{\ddagger}", embeddings.bert_ga),
        ("mBERT", embeddings.bert_m),
        ("mBERT_{\ddagger}", embeddings.bert_ma),
        ("BERT_{ner}", embeddings.bert_ner),
    ]:
        for suffix, kwargs in [
            ("", {"add_adj": False}),
            (" + ADJ", {"add_adj": True}),
        ]:
            approach = embedding + suffix
            logging.info(approach)
            scores = list()
            for novel in data.values():
                X, y = embeddings.vectorize(novel, model, **kwargs)
                clustering = Clustering("kmeans", X, y)
                score = clustering.evaluate()
                scores.append(score)
            scores = pd.DataFrame(scores)
            values = {
                index: f"{mean} (±{std})"
                for index, mean, std in zip(
                    scores.columns, scores.mean().round(2), scores.std().round(2)
                )
            }
            stats.append(values)
            index.append(approach)
    return pd.DataFrame(stats, index=index)


def compare_algorithms(model_directory, embedding):
    stats = list()
    index = list()

    data = load_data()
    embeddings = Embeddings(model_directory, "gutenberg", load=embedding)

    for algorithm in {"kmeans", "spectral", "ward"}:
        scores = list()
        for novel in data.values():
            X, y = embeddings.vectorize(novel, embeddings.TODO)
            clustering = Clustering(algorithm, X, y)
            score = clustering.evaluate()
            scores.append(score)
        scores = pd.DataFrame(scores)
        values = {
            index: f"{mean} (±{std})"
            for index, mean, std in zip(
                scores.columns, scores.mean().round(2), scores.std().round(2)
            )
        }
        stats.append(values)
        index.append(approach)
    return pd.DataFrame(stats, index=index)
