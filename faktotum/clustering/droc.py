from faktotum.clustering.utils import Embeddings, Clustering
from pathlib import Path
import pandas as pd

def load_data():
    package_folder = Path(__file__).parent.parent
    data_folder = Path(PACKAGE_FOLDER, "data", "droc", "linking")
    data = dict()
    for file_ in data_folder.glob("*.txt"):
        with file_.open("r", encoding="utf-8") as file_:
            data.update(json.load(file_))
    return data


def compare_embeddings(model_directory):
    stats = list()

    data = load_data()
    embeddings = Embeddings(model_directory, "gutenberg")

    for approach, model in {
        ("CBOW_{w2v}", embeddings.cbow_w2v),
        ("Skipgram_{w2v}", embeddings.skipgram_w2v),
        ("CBOW_{ft}", embeddings.cbow_ft),
        ("Skipgram_{ft}", embeddings.skipgram_ft),
        ("dBERT", embeddings.bert_g),
        ("dBERT_{\ddagger}", embeddings.bert_ga),
        ("mBERT", embeddings.bert_m),
        ("mBERT_{\ddagger}", embeddings.bert_ma),
        ("BERT_{ner}", embeddings.bert_ner),
    }:
        scores = list()
        for novel in data.values():
            X, y = embeddings.vectorize(
                novel, model, add_adj=False, add_nn=False, add_per=False
            )
            clustering = Clustering("kmeans", X, y)
            score = clustering.evaluate()
            scores.append(score)
        scores = pd.DataFrame(scores)
        stats.append({"approach": approach, "mean": scores.mean(), "std": scores.std()})
