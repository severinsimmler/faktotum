import logging
import random
from pathlib import Path

import flair
import torch
import pandas as pd
import scipy.cluster

flair.device = torch.device("cpu")
import numpy as np
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

logger = logging.getLogger("gensim")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("transformers")
logger.setLevel(logging.ERROR)

logging.basicConfig(format="%(message)s", level=logging.INFO)


class Embeddings:
    def __init__(self, model_directory, corpus, load="all"):
        if load in {"cbow-word2vec", "all"}:
            path = str(Path(model_directory, f"{corpus}-cbow.word2vec"))
            logging.info(f"Loading {path}...")
            self.cbow_w2v = Word2Vec.load(path)

        if load in {"skipgram-word2vec", "all"}:
            path = str(Path(model_directory, f"{corpus}-skipgram.word2vec"))
            logging.info(f"Loading {path}...")
            self.skipgram_w2v = Word2Vec.load(path)

        if load in {"cbow-fasttext-facebook", "all"}:
            path = str(Path(model_directory, f"facebook-{corpus}-cbow.fasttext"))
            logging.info(f"Loading {path}...")
            self.cbow_ft_fb = FastText.load(path)

        if load in {"cbow-fasttext", "all"}:
            path = str(Path(model_directory, f"{corpus}-cbow.fasttext"))
            logging.info(f"Loading {path}...")
            self.cbow_ft = FastText.load(path)

        if load in {"skipgram-fasttext", "all"}:
            path = str(Path(model_directory, f"{corpus}-skipgram.fasttext"))
            logging.info(f"Loading {path}...")
            self.skipgram_ft = FastText.load(path)

        if load in {"bert-german", "all"}:
            path = "bert-base-german-dbmdz-cased"
            logging.info(f"Loading {path}...")
            self.bert_g = BertEmbeddings(path)

        if load in {"bert-german-adapted", "all"}:
            path = str(Path(model_directory, f"bert-german-{corpus}-adapted"))
            logging.info(f"Loading {path}...")
            self.bert_ga = BertEmbeddings(path)

        if load in {"bert-multi", "all"}:
            path = "bert-base-multilingual-cased"
            logging.info(f"Loading {path}...")
            self.bert_m = BertEmbeddings(path)

        if load in {"bert-multi-adapted", "all"}:
            path = str(Path(model_directory, f"bert-multi-{corpus}-adapted"))
            logging.info(f"Loading {path}...")
            self.bert_ma = BertEmbeddings(path)

        if load in {"ner", "all"}:
            if corpus == "gutenberg":
                path = str(Path(model_directory, "ner-droc"))
            else:
                path = str(Path(model_directory, "ner-smartdata"))
            logging.info(f"Loading {path}...")
            self.bert_ner = BertEmbeddings(path)

    def vectorize(self, sentences, model, add_adj=False, add_nn=False, add_per=False):
        X = list()
        y = list()
        if isinstance(model, BertEmbeddings):
            _vectorize = self._bert_vectorization
        else:
            _vectorize = self._classic_vectorization
        for sentence in sentences:
            persons = list(self._group_persons(sentence))
            for identifier, indices in persons:
                vector = _vectorize(sentence, indices, model, add_adj, add_nn, add_per)
                X.append(vector)
                y.append(identifier)
        return np.array(X), np.array(y)

    def _classic_vectorization(
        self, sentence, token_indices, model, add_adj=False, add_nn=False, add_per=False
    ):
        self._add_tokens(sentence, token_indices, add_adj, add_nn, add_per)
        tokens = [token[0] for i, token in enumerate(sentence) if i in token_indices]
        return sum(self._get_classic_embedding(tokens, model)) / len(tokens)

    def _bert_vectorization(
        self, sentence, token_indices, model, add_adj=False, add_nn=False, add_per=False
    ):
        text = " ".join(token[0] for token in sentence)
        sentence_ = Sentence(text, use_tokenizer=False)
        model.embed(sentence_)
        self._add_tokens(sentence, token_indices, add_adj, add_nn, add_per)
        tokens = [token for i, token in enumerate(sentence_) if i in token_indices]
        return sum(self._get_bert_embedding(tokens)) / len(tokens)

    @staticmethod
    def _group_persons(sentence):
        indices = list()
        current_person = None
        last_index = 0
        for i, token in enumerate(sentence):
            if token[2] != "-":
                if token[2] == current_person and i - 1 == last_index:
                    indices.append(i)
                else:
                    if indices:
                        yield current_person, indices
                    indices = list()
                    indices.append(i)
                    current_person = token[2]
                    last_index = i
        if indices:
            yield current_person, indices

    @staticmethod
    def _add_tokens(sentence, token_indices, add_adj, add_nn, add_per):
        if add_adj:
            adjs = [i for i, token in enumerate(sentence) if "ADJA" in token[3]]
            token_indices.extend(adjs)

    @staticmethod
    def _get_bert_embedding(tokens):
        for token in tokens:
            yield token.get_embedding().numpy()

    @staticmethod
    def _get_classic_embedding(tokens, model):
        for token in tokens:
            try:
                yield model.wv[token]
            except KeyError:
                # Yield a null vector if not in vocabulary
                yield np.array([0] * 300)

class SemiSupervisedKMeans:
    def __init__(self, n_clusters, y, random_state=23):
        self.n_clusters = n_clusters
        self.y = y
        random.seed(random_state)

    def fit_predict(self, X):
        centroids = np.array(list(self._calculate_centroids(X)))
        _, y = scipy.cluster.vq.kmeans2(centroids, self.n_clusters, minit='matrix')
        return y

    def _calculate_centroids(self, X):
        X = pd.DataFrame(X)
        X["y"] = self.y
        for _, cluster in X.groupby("y"):
            yield random.choice(X.iloc[:, :-1].values)

class Clustering:
    def __init__(self, algorithm, X, y):
        algorithm = self.algorithms.get(algorithm, KMeans)
        self.X = X
        self.y = y
        self.n_clusters = len(set(y))
        self.random_state = 23
        if algorithm is SemiSupervisedKMeans:
            self.model = algorithm(n_clusters=self.n_clusters, y=y, random_state=self.random_state)
        else:
            self.model = algorithm(
                n_clusters=self.n_clusters, random_state=self.random_state, n_jobs=-1
            )

    @property
    def algorithms(self):
        return {
            "kmeans": KMeans,
            "semi-supervised-kmeans": SemiSupervisedKMeans,
            "spectral": SpectralClustering,
            "ward": AgglomerativeClustering,
        }

    def evaluate(self):
        y_ = self.model.fit_predict(self.X)
        homogeneity = metrics.homogeneity_score(self.y, y_)
        completeness = metrics.completeness_score(self.y, y_)
        v_measure = metrics.v_measure_score(self.y, y_)
        ari = metrics.adjusted_rand_score(self.y, y_)
        ami = metrics.adjusted_mutual_info_score(self.y, y_)
        fmi = metrics.fowlkes_mallows_score(self.y, y_)
        return {
            "Homogeneity": round(homogeneity, 2),
            "Completeness": round(completeness, 2),
            "V-Measure": round(v_measure, 2),
            "ARI": round(ari, 2),
            "AMI": round(ami, 2),
            "FMI": round(fmi, 2),
        }


def compare_approaches_global(data, model_directory, corpus):
    logging.info(TABLE_BEGIN)

    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    result = word2vec(str(path), data)
    result["approach"] = "CBOW\\textsubscript{w2v}"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    result = word2vec(str(path), data)
    result["approach"] = "Skipgram\\textsubscript{w2v}"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    result = fasttext(str(path), data)
    result["approach"] = "CBOW\\textsubscript{ft}"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    result = fasttext(str(path), data)
    result["approach"] = "Skipgram\\textsubscript{ft}"
    logging.info(TABLE_ROW.format(**result))

    ##########

    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    result = word2vec(str(path), data, add_adj=True)
    result["approach"] = "CBOW\\textsubscript{w2v} + ADJ"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    result = word2vec(str(path), data, add_adj=True)
    result["approach"] = "Skipgram\\textsubscript{w2v} + ADJ"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    result = fasttext(str(path), data, add_adj=True)
    result["approach"] = "CBOW\\textsubscript{ft} + ADJ"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    result = fasttext(str(path), data, add_adj=True)
    result["approach"] = "Skipgram\\textsubscript{ft} + ADJ"
    logging.info(TABLE_ROW.format(**result))

    ###################

    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    result = word2vec(str(path), data, add_per=True)
    result["approach"] = "CBOW\\textsubscript{w2v} + PER"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    result = word2vec(str(path), data, add_per=True)
    result["approach"] = "Skipgram\\textsubscript{w2v} + PER"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    result = fasttext(str(path), data, add_per=True)
    result["approach"] = "CBOW\\textsubscript{ft} + PER"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    result = fasttext(str(path), data, add_per=True)
    result["approach"] = "Skipgram\\textsubscript{ft} + PER"
    logging.info(TABLE_ROW.format(**result))

    #######################

    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    result = word2vec(str(path), data, add_adj=True, add_per=True)
    result["approach"] = "CBOW\\textsubscript{w2v} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    result = word2vec(str(path), data, add_adj=True, add_per=True)
    result["approach"] = "Skipgram\\textsubscript{w2v} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    result = fasttext(str(path), data, add_adj=True, add_per=True)
    result["approach"] = "CBOW\\textsubscript{ft} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    result = fasttext(str(path), data, add_adj=True, add_per=True)
    result["approach"] = "Skipgram\\textsubscript{w2v} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    #################

    result = bert("bert-base-german-dbmdz-cased", data)
    result["approach"] = "dBERT"
    logging.info(TABLE_ROW.format(**result))

    result = bert("bert-base-multilingual-cased", data)
    result["approach"] = "mBERT"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-german-gutenberg-adapted"))
    else:
        path = str(Path(model_directory, "bert-german-presse-adapted"))
    result = bert(path, data)
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$}"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        path = str(Path(model_directory, "bert-multi-presse-adapted"))
    result = bert(path, data)
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$}"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "ner-droc"))
    else:
        path = str(Path(model_directory, "ner-presse"))
    result = bert(path, data)
    result["approach"] = "glBERT"
    logging.info(TABLE_ROW.format(**result))

    ###############################

    result = bert("bert-base-german-dbmdz-cased", data, add_adj=True)
    result["approach"] = "dBERT + ADJ"
    logging.info(TABLE_ROW.format(**result))

    result = bert("bert-base-multilingual-cased", data, add_adj=True)
    result["approach"] = "mBERT + ADJ"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-german-gutenberg-adapted"))
    else:
        path = str(Path(model_directory, "bert-german-presse-adapted"))
    result = bert(path, data, add_adj=True)
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + ADJ"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        path = str(Path(model_directory, "bert-multi-presse-adapted"))
    result = bert(path, data, add_adj=True)
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + ADJ"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "ner-droc"))
    else:
        path = str(Path(model_directory, "ner-presse"))
    result = bert(path, data, add_adj=True)
    result["approach"] = "glBERT + ADJ"
    logging.info(TABLE_ROW.format(**result))

    #################################

    result = bert("bert-base-german-dbmdz-cased", data, add_per=True)
    result["approach"] = "dBERT + PER"
    logging.info(TABLE_ROW.format(**result))

    result = bert("bert-base-multilingual-cased", data, add_per=True)
    result["approach"] = "mBERT + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-german-gutenberg-adapted"))
    else:
        path = str(Path(model_directory, "bert-german-presse-adapted"))
    result = bert(path, data, add_per=True)
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        path = str(Path(model_directory, "bert-multi-presse-adapted"))
    result = bert(path, data, add_per=True)
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "ner-droc"))
    else:
        path = str(Path(model_directory, "ner-presse"))
    result = bert(path, data, add_per=True)
    result["approach"] = "glBERT + PER"
    logging.info(TABLE_ROW.format(**result))

    ###########################################

    result = bert("bert-base-german-dbmdz-cased", data, add_adj=True, add_per=True)
    result["approach"] = "dBERT + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    result = bert("bert-base-multilingual-cased", data, add_adj=True, add_per=True)
    result["approach"] = "mBERT + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-german-gutenberg-adapted"))
    else:
        path = str(Path(model_directory, "bert-german-presse-adapted"))
    result = bert(path, data, add_adj=True, add_per=True)
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        path = str(Path(model_directory, "bert-multi-presse-adapted"))
    result = bert(path, data, add_adj=True, add_per=True)
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "ner-droc"))
    else:
        path = str(Path(model_directory, "ner-presse"))
    result = bert(path, data, add_adj=True, add_per=True)
    result["approach"] = "glBERT + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    #############################

    if corpus == "gutenberg":
        bert_path = "bert-base-multilingual-cased"
        classic_path = Path(model_directory, f"{corpus}-cbow.fasttext")
    else:
        bert_path = "bert-base-multilingual-cased"
    result = stacked(bert_path, str(classic_path), data)
    result["approach"] = "mBERT + CBOW\\textsubscript{ft} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    logging.info(TABLE_END)


class LocalKMeans:
    def __init__(self, dataset, model_directory):
        self.dataset = dataset
        model_directory

    def word2vec(self, cbow: bool = True):
        if cbow:
            path = Path(model_directory, "gutenberg-cbow.word2vec")
            approach = "CBOW_{w2v}"
        else:
            path = Path(model_directory, "gutenberg-skipgram.word2vec")
            approach = "Skipgram_{w2v}"

        for novel in self.dataset.values():
            scores = word2vec(path, novel)


def compare_approaches_local(data, model_directory, corpus):
    logging.info(TABLE_BEGIN)

    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    results = list()
    for doc in data.values():
        result = word2vec(str(path), doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "CBOW\\textsubscript{w2v}"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    results = list()
    for doc in data.values():
        result = word2vec(str(path), doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "Skipgram\\textsubscript{w2v}"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    results = list()
    for doc in data.values():
        result = fasttext(str(path), doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "CBOW\\textsubscript{ft}"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    results = list()
    for doc in data.values():
        result = fasttext(str(path), doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "Skipgram\\textsubscript{ft}"
    logging.info(TABLE_ROW_STD.format(**result))

    ##########

    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    results = list()
    for doc in data.values():
        result = word2vec(str(path), doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "CBOW\\textsubscript{w2v} + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    results = list()
    for doc in data.values():
        result = word2vec(str(path), doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "Skipgram\\textsubscript{w2v} + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    results = list()
    for doc in data.values():
        result = fasttext(str(path), doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "CBOW\\textsubscript{ft} + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    results = list()
    for doc in data.values():
        result = fasttext(str(path), doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "Skipgram\\textsubscript{ft} + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    ###################

    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    results = list()
    for doc in data.values():
        result = word2vec(str(path), doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "CBOW\\textsubscript{w2v} + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    results = list()
    for doc in data.values():
        result = word2vec(str(path), doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "Skipgram\\textsubscript{w2v} + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    results = list()
    for doc in data.values():
        result = fasttext(str(path), doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "CBOW\\textsubscript{ft} + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    results = list()
    for doc in data.values():
        result = fasttext(str(path), doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "Skipgram\\textsubscript{ft} + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    #######################

    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    results = list()
    for doc in data.values():
        result = word2vec(str(path), doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "CBOW\\textsubscript{w2v} + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    results = list()
    for doc in data.values():
        result = word2vec(str(path), doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "Skipgram\\textsubscript{w2v} + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    results = list()
    for doc in data.values():
        result = fasttext(str(path), doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "CBOW\\textsubscript{ft} + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    results = list()
    for doc in data.values():
        result = fasttext(str(path), doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "Skipgram\\textsubscript{w2v} + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    #################

    results = list()
    for doc in data.values():
        result = bert("bert-base-german-dbmdz-cased", doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "dBERT"
    logging.info(TABLE_ROW_STD.format(**result))

    results = list()
    for doc in data.values():
        result = bert("bert-base-multilingual-cased", doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "mBERT"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-german-gutenberg-adapted"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$}"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$}"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "ner-droc"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "glBERT"
    logging.info(TABLE_ROW_STD.format(**result))

    ###############################

    results = list()
    for doc in data.values():
        result = bert("bert-base-german-dbmdz-cased", doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "dBERT + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    results = list()
    for doc in data.values():
        result = bert("bert-base-multilingual-cased", doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "mBERT + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-german-gutenberg-adapted"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "ner-droc"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_adj=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "glBERT + ADJ"
    logging.info(TABLE_ROW_STD.format(**result))

    #################################

    results = list()
    for doc in data.values():
        result = bert("bert-base-german-dbmdz-cased", doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "dBERT + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    results = list()
    for doc in data.values():
        result = bert("bert-base-multilingual-cased", doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "mBERT + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-german-gutenberg-adapted"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "ner-droc"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "glBERT + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    ###########################################

    results = list()
    for doc in data.values():
        result = bert("bert-base-german-dbmdz-cased", doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "dBERT + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    results = list()
    for doc in data.values():
        result = bert("bert-base-multilingual-cased", doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "mBERT + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-german-gutenberg-adapted"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "ner-droc"))
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = bert(path, doc, add_adj=True, add_per=True)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]
    result["approach"] = "glBERT + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    #############################
    return
    if corpus == "gutenberg":
        bert_path = "bert-base-multilingual-cased"
        classic_path = Path(model_directory, f"{corpus}-cbow.fasttext")
    else:
        raise NotImplementedError
    results = list()
    for doc in data.values():
        result = stacked(bert_path, str(classic_path), data)
        results.append(result)
    df = pd.DataFrame(results)
    result = df.mean().round(2).to_dict()
    std = df.std().round(2)
    result["std_homogeneity"] = std["homogeneity"]
    result["std_completeness"] = std["completeness"]
    result["std_v"] = std["v"]

    result["approach"] = "mBERT + CBOW\\textsubscript{ft} + ADJ + PER"
    logging.info(TABLE_ROW_STD.format(**result))

    logging.info(TABLE_END)
