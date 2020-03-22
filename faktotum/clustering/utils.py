import logging
import json
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

        if load in {"all-masked", "all"}:
            if corpus == "gutenberg":
                path = str(Path(model_directory, "entity-embeddings-droc-all-masked"))
            else:
                path = str(
                    Path(model_directory, "entity-embeddings-smartdata-all-masked")
                )
            logging.info(f"Loading {path}...")
            self.entity_all_bert = BertEmbeddings(path)

        if load in {"entity", "all"}:
            if corpus == "gutenberg":
                path = str(Path(model_directory, "entity-embeddings-droc"))
            else:
                path = str(Path(model_directory, "entity-embeddings-smartdata"))
            logging.info(f"Loading {path}...")
            self.entity_bert = BertEmbeddings(path)

    def vectorize(self, sentences, model, add_adj=False, return_str=False):
        X = list()
        y = list()
        strs = list()
        if isinstance(model, BertEmbeddings):
            _vectorize = self._bert_vectorization
        else:
            _vectorize = self._classic_vectorization
        for sentence in sentences:
            persons = list(self._group_persons(sentence))
            for identifier, indices in persons:
                person = list()
                for index in indices:
                    person.append(sentence[index])
                strs.append({"sentence": sentence, "index": indices})
                vector = _vectorize(sentence, indices, model, add_adj)
                X.append(vector)
                y.append(identifier)
        if return_str:
            return np.array(X), np.array(y), strs
        else:
            return np.array(X), np.array(y)

    def _classic_vectorization(self, sentence, token_indices, model, add_adj=False):
        self._add_tokens(sentence, token_indices, add_adj)
        tokens = [token[0] for i, token in enumerate(sentence) if i in token_indices]
        return sum(self._get_classic_embedding(tokens, model)) / len(tokens)

    def _bert_vectorization(self, sentence, token_indices, model, add_adj=False):
        text = " ".join(token[0] for token in sentence)
        sentence_ = Sentence(text, use_tokenizer=False)
        model.embed(sentence_)
        self._add_tokens(sentence, token_indices, add_adj)
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
    def _add_tokens(sentence, token_indices, add_adj):
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
    def __init__(self, y, random_state=23):
        self.y = y
        random.seed(random_state)

    def fit_predict(self, X):
        centroids = np.array(list(self._calculate_centroids(X)))
        _, y = scipy.cluster.vq.kmeans2(X, centroids, minit="matrix")
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
            self.model = algorithm(self.y, random_state=self.random_state)
        elif algorithm is AgglomerativeClustering:
            self.model = self.model = algorithm(n_clusters=self.n_clusters)
        else:
            self.model = algorithm(
                n_clusters=self.n_clusters, random_state=self.random_state, n_jobs=-1
            )

    @property
    def algorithms(self):
        return {
            "kmeans": KMeans,
            "semi-supervised-kmeans": SemiSupervisedKMeans,
            "ward": AgglomerativeClustering,
        }

    def evaluate(self, i=None, strs=None):
        y_ = self.model.fit_predict(self.X)
        homogeneity = metrics.homogeneity_score(self.y, y_)
        completeness = metrics.completeness_score(self.y, y_)
        v_measure = metrics.v_measure_score(self.y, y_)
        ari = metrics.adjusted_rand_score(self.y, y_)
        ami = metrics.adjusted_mutual_info_score(self.y, y_)
        fmi = metrics.fowlkes_mallows_score(self.y, y_)
        if strs:
            with open(f"ward-smartdata-{i}.json", "w", encoding="utf-8") as f:
                data = {"gold": list(self.y), "pred": [int(x) for x in y_], "str": strs}
                json.dump(data, f, ensure_ascii=False, indent=4)
        return {
            "Homogeneity": round(homogeneity, 2),
            "Completeness": round(completeness, 2),
            "V-Measure": round(v_measure, 2),
            "ARI": round(ari, 2),
            "AMI": round(ami, 2),
            "FMI": round(fmi, 2),
        }
