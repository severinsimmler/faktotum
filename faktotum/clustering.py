from gensim.models.fasttext import fasttext
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from flair.embeddings import BertEmbeddings
from flair.data import Sentence
def classic_vectorization(mentions, model):
    for mention in mentions:
        try:
            vector = model.wv[mention["mention"]]
        except KeyError:
            # return null vector if not in vocabulary (word2vec)
            vector = [0] * 300
        yield mention["id"], vector


def bert_vectorization(mentions, model):
    for mention in mentions:
        text = " ".join([token[0] for token in mention["sentence"]])
        sentence = Sentence(text, use_tokenizer=False)
        model.embed(sentence)
        yield mention["id"], sentence[mention["index"]].get_embedding().numpy()


def word2vec(modelpath, data):
    model = Word2Vec.load(modelpath)
    distinct_classes = set([mention["id"] for mention in data])
    classes = dict(enumerate(distinct_classes))
    labels_true = list()
    vectors = list()

    for i, vector in classic_vectorization(data, model):
        labels_true.append(classes[i])
        vectors.append(vector)

    X = np.array(values)
    labels_pred = KMeans(n_clusters=len(classes), random_state=23).fit_predict(X)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    return {"homogeneity": homogeneity, "completeness": completeness, "v": v}

def fasttext(modelpath, data):
    model = FastText.load(modelpath)
    distinct_classes = set([mention["id"] for mention in data])
    classes = dict(enumerate(distinct_classes))
    labels_true = list()
    vectors = list()

    for i, vector in classic_vectorization(data, model):
        labels_true.append(classes[i])
        vectors.append(vector)

    X = np.array(values)
    labels_pred = KMeans(n_clusters=len(classes), random_state=23).fit_predict(X)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    return {"homogeneity": homogeneity, "completeness": completeness, "v": v}



def bert(modelpath, data):
    model = BertEmbeddings(modelpath)
    distinct_classes = set([mention["id"] for mention in data])
    classes = dict(enumerate(distinct_classes))
    labels_true = list()
    vectors = list()

    for i, vector in bert_vectorization(data, model):
        labels_true.append(classes[i])
        vectors.append(vector)

    X = np.array(values)
    labels_pred = KMeans(n_clusters=len(classes), random_state=23).fit_predict(X)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    return {"homogeneity": homogeneity, "completeness": completeness, "v": v}
