import numpy as np
import pandas as pd
import tqdm
import transformers

from faktotum.utils import sentencize, MODELS, pool_entity, extract_features
from faktotum.kb import KnowledgeBase
from faktotum.typing import Entities, KnowledgeBase, Pipeline, TaggedTokens


def nel(text: str, kb: KnowledgeBase, domain: str = "literary-texts"):
    tagged_tokens = ner(text, domain)
    return ned(tagged_tokens, kb, domain)


def ner(text: str, domain: str = "literary-texts"):
    model_name = MODELS["ner"][domain]
    pipeline = transformers.pipeline(
        "ner", model=model_name, tokenizer=model_name, ignore_labels=[]
    )
    sentences = [(i, sentence) for i, sentence in enumerate(sentencize(text))]
    predictions = list()
    for i, sentence in tqdm.tqdm(sentences):
        sentence = "".join(str(token) for token in sentence)
        prediction = _predict_labels(pipeline, sentence, i)
        predictions.extend(prediction)
    return pd.DataFrame(predictions).loc[:, ["sentence_id", "word", "entity"]]


def ned(tokens: TaggedTokens, kb: KnowledgeBase = None, domain: str = "literary-texts"):
    model_name = MODELS["ned"][domain]
    pipeline = transformers.pipeline(
        "feature-extraction", model=model_name, tokenizer=model_name
    )
    identifiers = list()
    for sentence_id, sentence in tokens.groupby("sentence_id"):
        entities = sentence.dropna()
        features = extract_features(pipeline, sentence.loc[:, "word"])
        for indices, mention in _group_mentions(entities):
            vector = pool_entity(mention, features)
            best_candidate, score = _get_best_candidate(vector, kb)
            identifiers.append((indices, best_candidate))
    tokens["entity_id"] = np.nan
    for mention, candidate in identifiers:
        tokens.iloc[mention, -1] = candidate
    return tokens


def _predict_labels(pipeline: Pipeline, sentence: str, sentence_id: int) -> Entities:
    entities = list()
    for token in pipeline(sentence):
        if token["word"] not in {"[CLS]", "[SEP]", "[MASK]"}:
            token["sentence_id"] = sentence_id
            if token["word"].startswith("##"):
                entities[-1]["word"] += token["word"][2:]
            else:
                del token["score"]
                if token["entity"] == "O":
                    token["entity"] = np.nan
                entities.append(token)
    return entities


def _get_best_candidate(vector, kb):
    best_candidate = "NIL"
    best_score = 0.0
    for identifier, values in kb.items():
        for candidate in values["EMBEDDINGS"]:
            score = _cosine_similarity(vector, candidate)
            if score > best_score:
                best_score = score
                best_candidate = identifier
    return best_candidate, best_score


def _cosine_similarity(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))


def _group_mentions(entities):
    mention = list()
    indices = list()
    rows = entities.reset_index().iterrows()
    for index, (row, token) in zip(entities.index, rows):
        if token["entity"].startswith("B"):
            if mention:
                yield indices, mention
                mention = list()
                indices = list()
            mention.append(row)
            indices.append(index)
        elif token["entity"].startswith("I"):
            if mention:
                if mention[-1] == row - 1:
                    mention.append(row)
                    indices.append(index)
    if mention:
        yield indices, mention
