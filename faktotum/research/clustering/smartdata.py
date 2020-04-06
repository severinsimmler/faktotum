import json
import logging
from pathlib import Path

import pandas as pd

from faktotum.research.clustering.utils import Clustering, Embeddings

logging.basicConfig(format="%(message)s", level=logging.INFO)


def load_data(all_=True):
    sentence = list()
    package_folder = Path(__file__).parent.parent
    data_folder = Path(package_folder, "data", "smartdata", "linking")
    data = dict()
    if all_:
        files = [
            Path(data_folder, "train.txt"),
            Path(data_folder, "test.txt"),
            Path(package_folder, "data", "wikidata.txt"),
        ]
    else:
        files = [
            Path(data_folder, "test.txt"),
            Path(package_folder, "data", "wikidata-test.txt"),
        ]
    for file_ in files:
        text = file_.read_text(encoding="utf-8")
        for line in text.split("\n"):
            if not line.startswith("#"):
                if line != "":
                    sentence.append(line.split(" "))
                else:
                    yield sentence
                    sentence = list()
        if sentence:
            yield sentence


def ward(model_directory):
    stats = list()
    index = list()

    data = list(load_data(all_=False))
    embeddings = Embeddings(model_directory, "presse", load="all")
    for name, e in [
        ("vanilla", embeddings.bert_ma),
        ("all", embeddings.entity_all_bert),
        ("random", embeddings.entity_bert),
    ]:
        X, y, strs = embeddings.vectorize(data, e, return_str=True)
        print(X)
        print(X.shape)
        clustering = Clustering("ward", X, y)
        score = clustering.evaluate(strs=strs, i=name)
        print(name)
        print(score)
        print()
    return score


def compare_embeddings(model_directory):
    stats = list()
    index = list()

    data = list(load_data())
    embeddings = Embeddings(model_directory, "presse")

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
            X, y = embeddings.vectorize(data, model, **kwargs)
            clustering = Clustering("kmeans", X, y)
            score = clustering.evaluate()
            stats.append(score)
            index.append(approach)
    return pd.DataFrame(stats, index=index)


def compare_algorithms(model_directory, embedding):
    stats = list()
    index = list()

    data = list(load_data(all_=False))
    embeddings = Embeddings(model_directory, "presse", load=embedding)

    X, y = embeddings.vectorize(data, embeddings.bert_ma)
    for algorithm in {"kmeans", "ward", "semi-supervised-kmeans"}:
        scores = list()
        clustering = Clustering(algorithm, X, y)
        score = clustering.evaluate()
        stats.append(score)
        index.append(algorithm)
    return pd.DataFrame(stats, index=index)
