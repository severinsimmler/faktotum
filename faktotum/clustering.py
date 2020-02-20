from pathlib import Path

import numpy as np
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from sklearn import metrics
from sklearn.cluster import KMeans


def classic_vectorization(mentions, model, add_adj=False, add_per=False):
    for mention in mentions:
        try:
            vector = model.wv[mention["mention"]]
        except KeyError:
            # return null vector if not in vocabulary (word2vec)
            vector = [0] * 300
        if add_adj:
            adjs = [token[0] for token in mention["sentence"] if token[3] == "ADJA"]
            try:
                for adj in adjs:
                    vector = vector + model.wv[adj]
            except KeyError:
                pass
        if add_per:
            pers = [token[0] for token in mention["sentence"] if "PER" in token[1]]
            try:
                for per in pers:
                    vector = vector + model.wv[per]
            except KeyError:
                pass
        yield mention["id"], vector


def bert_vectorization(mentions, model):
    for mention in mentions:
        text = " ".join([token[0] for token in mention["sentence"]])
        sentence = Sentence(text, use_tokenizer=False)
        model.embed(sentence)
        yield mention["id"], sentence[mention["index"]].get_embedding().numpy()


def word2vec(modelpath, data, add_adj=False, add_per=False):
    model = Word2Vec.load(modelpath)
    distinct_classes = set([mention["id"] for mention in data])
    classes = {c: i for i, c in enumerate(distinct_classes)}
    labels_true = list()
    vectors = list()

    for i, vector in classic_vectorization(data, model, add_adj, add_per):
        labels_true.append(classes[i])
        vectors.append(vector)

    X = np.array(vectors)
    labels_pred = KMeans(n_clusters=len(classes), random_state=23).fit_predict(X)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(
        labels_true, labels_pred
    )
    return {"homogeneity": homogeneity, "completeness": completeness, "v": v}


def fasttext(modelpath, data, add_adj=False, add_per=False):
    model = FastText.load(modelpath)
    distinct_classes = set([mention["id"] for mention in data])
    classes = {c: i for i, c in enumerate(distinct_classes)}
    labels_true = list()
    vectors = list()

    for i, vector in classic_vectorization(data, model, add_adj, add_per):
        labels_true.append(classes[i])
        vectors.append(vector)

    X = np.array(vectors)
    labels_pred = KMeans(n_clusters=len(classes), random_state=23).fit_predict(X)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(
        labels_true, labels_pred
    )
    return {"homogeneity": homogeneity, "completeness": completeness, "v": v}


def bert(modelpath, data):
    model = BertEmbeddings(modelpath)
    distinct_classes = set([mention["id"] for mention in data])
    classes = {c: i for i, c in enumerate(distinct_classes)}
    labels_true = list()
    vectors = list()

    for i, vector in bert_vectorization(data, model):
        labels_true.append(classes[i])
        vectors.append(vector)

    X = np.array(vectors)
    labels_pred = KMeans(n_clusters=len(classes), random_state=23).fit_predict(X)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(
        labels_true, labels_pred
    )
    return {"homogeneity": homogeneity, "completeness": completeness, "v": v}


def compare_approaches(data, model_directory, corpus):
    print("CBOW Word2Vec")
    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    print(word2vec(str(path), data), "\n")

    print("Skipgram Word2Vec")
    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    print(word2vec(str(path), data), "\n")

    print("CBOW FastText")
    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    print(fasttext(str(path), data), "\n")

    print("Skipgram FastText")
    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    print(fasttext(str(path), data), "\n")

    ##########

    print("CBOW Word2Vec + ADJ")
    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    print(word2vec(str(path), data, add_adj=True), "\n")

    print("Skipgram Word2Vec + ADJ")
    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    print(word2vec(str(path), data, add_adj=True), "\n")

    print("CBOW FastText + ADJ")
    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    print(fasttext(str(path), data, add_adj=True), "\n")

    print("Skipgram FastText + ADJ")
    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    print(fasttext(str(path), data, add_adj=True), "\n")


    ###################


    print("CBOW Word2Vec + PER")
    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    print(word2vec(str(path), data, add_per=True), "\n")

    print("Skipgram Word2Vec + PER")
    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    print(word2vec(str(path), data, add_per=True), "\n")

    print("CBOW FastText + PER")
    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    print(fasttext(str(path), data, add_per=True), "\n")

    print("Skipgram FastText + PER")
    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    print(fasttext(str(path), data, add_per=True), "\n")

    #######################

    print("CBOW Word2Vec + ADJ + PER")
    path = Path(model_directory, f"{corpus}-cbow.word2vec")
    print(word2vec(str(path), data, add_adj=True, add_per=True), "\n")

    print("Skipgram Word2Vec + ADJ + PER")
    path = Path(model_directory, f"{corpus}-skipgram.word2vec")
    print(word2vec(str(path), data, add_adj=True, add_per=True), "\n")

    print("CBOW FastText + ADJ + PER")
    path = Path(model_directory, f"{corpus}-cbow.fasttext")
    print(fasttext(str(path), data, add_adj=True, add_per=True), "\n")

    print("Skipgram FastText + ADJ + PER")
    path = Path(model_directory, f"{corpus}-skipgram.fasttext")
    print(fasttext(str(path), data, add_adj=True, add_per=True), "\n")

    #################

    print("German vanilla BERT")
    print(bert("bert-base-german-dbmdz-cased", data), "\n")

    print("Multi vanilla BERT")
    print(bert("bert-base-multilingual-cased", data), "\n")

    print("Adapted German BERT")
    if corpus == "gutenberg":
        path = Path(model_directory, "bert-german-literary-adapted")
    else:
        raise NotImplementedError
    print(bert(path, data), "\n")

    print("Adapted Multi BERT")
    if corpus == "gutenberg":
        path = Path(model_directory, "bert-multi-literary-adapted")
    else:
        raise NotImplementedError
    print(bert(path, data), "\n")

    print("German NER trained BERT")
    if corpus == "gutenberg":
        path = Path(model_directory, "german-literary-bert")
    else:
        raise NotImplementedError
    print(bert(path, data), "\n")

    # TODO: stacked
    # TODO: BERT mit explizitem Kontext
