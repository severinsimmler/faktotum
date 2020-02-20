import logging
from pathlib import Path

import flair
import torch

flair.device = torch.device("cpu")
import numpy as np
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from sklearn import metrics
from sklearn.cluster import KMeans

logger = logging.getLogger("gensim")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("transformers")
logger.setLevel(logging.ERROR)

logging.basicConfig(format="%(message)s", level=logging.INFO)


TABLE_BEGIN = "\\begin{table}\n  \centering\n\\begin{tabular}{lllll}\n  \\toprule\n{} & Homogeneity & Completeness & V \\\\\n \\midrule"
TABLE_END = "\\bottomrule\n  \\end{tabular}\n\\caption{Caption}\n\\end{table}"
TABLE_ROW = "{approach} & {homogeneity} & {completeness} & {v} \\\\"


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


def bert_vectorization(mentions, model, add_adj=False, add_per=False):
    for mention in mentions:
        text = " ".join([token[0] for token in mention["sentence"]])
        sentence = Sentence(text, use_tokenizer=False)
        model.embed(sentence)
        vector = sentence[mention["index"]].get_embedding().numpy()
        if add_adj:
            adjs = [i for i, token in enumerate(mention["sentence"]) if token[3] == "ADJA"]
            for adj in adjs:
                vector = vector + sentence[adj].get_embedding().numpy()
        if add_per:
            pers = [i for i, token in enumerate(mention["sentence"]) if "PER" in token[1]]
            for per in pers:
                vector = vector + sentence[per].get_embedding().numpy()
        yield mention["id"], vector


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
    return {
        "homogeneity": round(homogeneity * 100, 2),
        "completeness": round(completeness * 100, 2),
        "v": round(v * 100, 2),
    }


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
    return {
        "homogeneity": round(homogeneity * 100, 2),
        "completeness": round(completeness * 100, 2),
        "v": round(v * 100, 2),
    }


def bert(modelpath, data, add_adj=False, add_per=False):
    model = BertEmbeddings(modelpath)
    distinct_classes = set([mention["id"] for mention in data])
    classes = {c: i for i, c in enumerate(distinct_classes)}
    labels_true = list()
    vectors = list()

    for i, vector in bert_vectorization(data, model, add_adj, add_per):
        labels_true.append(classes[i])
        vectors.append(vector)

    X = np.array(vectors)
    labels_pred = KMeans(n_clusters=len(classes), random_state=23).fit_predict(X)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(
        labels_true, labels_pred
    )
    return {
        "homogeneity": round(homogeneity * 100, 2),
        "completeness": round(completeness * 100, 2),
        "v": round(v * 100, 2),
    }


def stacked(bert_path, classic_path, data, add_adj=False, add_per=False):
    bert = BertEmbeddings(bert_path)
    classic = FastText.load(classic_path)
    distinct_classes = set([mention["id"] for mention in data])
    classes = {c: i for i, c in enumerate(distinct_classes)}
    labels_true = list()
    vectors = list()

    for i, bert_vector, classic_vector in zip(bert_vectorization(data, model, add_adj, add_per), classic_vectorization(data, model, add_adj, add_per)):
        labels_true.append(classes[i])
        vector = np.concatenate((bert_vector, classic_vector))
        vectors.append(vector)

    X = np.array(vectors)
    labels_pred = KMeans(n_clusters=len(classes), random_state=23).fit_predict(X)
    homogeneity, completeness, v = metrics.homogeneity_completeness_v_measure(
        labels_true, labels_pred
    )
    return {
        "homogeneity": round(homogeneity * 100, 2),
        "completeness": round(completeness * 100, 2),
        "v": round(v * 100, 2),
    }


def compare_approaches(data, model_directory, corpus):
    '''
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
    logging.info(TABLE_ROW.format(**result))path

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
        raise NotImplementedError
    result = bert(path, data)
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$}"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        raise NotImplementedError
    result = bert(path, data)
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$}"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "german-literary-bert"))
    else:
        raise NotImplementedError
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
        raise NotImplementedError
    result = bert(path, data, add_adj=True)
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + ADJ"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        raise NotImplementedError
    result = bert(path, data, add_adj=True)
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + ADJ"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "german-literary-bert"))
    else:
        raise NotImplementedError
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
        raise NotImplementedError
    result = bert(path, data, add_per=True)
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        raise NotImplementedError
    result = bert(path, data, add_per=True)
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "german-literary-bert"))
    else:
        raise NotImplementedError
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
        raise NotImplementedError
    result = bert(path, data, add_adj=True, add_per=True)
    result["approach"] = "dBERT\\textsuperscript{$\\ddagger$} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "bert-multi-gutenberg-adapted"))
    else:
        raise NotImplementedError
    result = bert(path, data, add_adj=True, add_per=True)
    result["approach"] = "mBERT\\textsuperscript{$\\ddagger$} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    if corpus == "gutenberg":
        path = str(Path(model_directory, "german-literary-bert"))
    else:
        raise NotImplementedError
    result = bert(path, data, add_adj=True, add_per=True)
    result["approach"] = "glBERT + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    #############################
    '''
    if corpus == "gutenberg":
        bert_path = "bert-base-multilingual-cased"
        classic_path = Path(model_directory, f"{corpus}-cbow.fasttext")
    else:
        raise NotImplementedError
    result = stacked(bert_path, classic_path, data)
    result["approach"] = "mBERT + CBOW\\textsubscript{ft} + ADJ + PER"
    logging.info(TABLE_ROW.format(**result))

    logging.info(TABLE_END)
