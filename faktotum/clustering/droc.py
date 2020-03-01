from faktotum.clustering.utils import Embeddings, Clustering
from pathlib import Path
import json
import pandas as pd
import logging


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

    for approach, model in {
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
    }:
        for suffix, features in [
            ("", {"add_adj": False, "add_nn": False, "add_per": False}),
            ("+ ADJ", {"add_adj": True, "add_nn": False, "add_per": False}),
            ("+ NN", {"add_adj": False, "add_nn": True, "add_per": False}),
            ("+ PER", {"add_adj": False, "add_nn": False, "add_per": True}),
            ("+ ADJ + NN + PER", {"add_adj": True, "add_nn": True, "add_per": True}),
        ]:
            approach = approach + suffix
            logging.info(approach)
            scores = list()
            for novel in data.values():
                X, y = embeddings.vectorize(novel, model, **features)
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
