import logging

import numpy as np
import pandas as pd
import tqdm
import transformers
from strsimpy.jaro_winkler import JaroWinkler

from faktotum.models import NamedEntityRecognition, NamedEntityDisambiguation
from faktotum.kb import KnowledgeBase
from faktotum.typing import Entities, Pipeline, TaggedTokens
from faktotum.utils import (
    align_index,
    extract_features,
    pool_tokens,
    sentencize,
)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


NER_MODELS = NamedEntityRecognition()
NED_MODELS = NamedEntityDisambiguation()
JARO_WINKLER = JaroWinkler()


def nel(
    text: str, kb: KnowledgeBase, domain: str = "literary-texts",
):
    tagged_tokens = ner(text, domain)
    return ned(tagged_tokens, kb, domain)


def ner(text: str, domain: str = "literary-texts"):
    pipeline = NER_MODEL[domain]
    sentences = [(i, sentence) for i, sentence in enumerate(sentencize(text))]
    predictions = list()
    logging.info("Processing sentences through NER pipeline...")
    for i, sentence in sentences:
        sentence = "".join(str(token) for token in sentence)
        prediction = _predict_labels(pipeline, sentence, i)
        predictions.extend(prediction)
    return pd.DataFrame(predictions).loc[:, ["sentence_id", "word", "entity"]]


def ned(
    tokens: TaggedTokens, kb: KnowledgeBase = None, domain: str = "literary-texts",
):
    pipeline = NED_MODEL[domain]
    identifiers = list()
    logging.info("Processing sentences through NED pipeline...")
    for sentence_id, sentence in tokens.groupby("sentence_id"):
        entities = sentence.dropna()
        index_mapping, features = extract_features(pipeline, sentence.loc[:, "word"])
        for original_index, index, mention in _group_mentions(entities):
            aligned_index = align_index(index, index_mapping)
            mention_embedding = pool_tokens(aligned_index, features)
            best_candidate, score = _get_best_candidate(mention, mention_embedding, kb)
            identifiers.append((original_index, best_candidate))
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


def _get_best_candidate(mention, mention_embedding, kb):
    best_candidate = "NIL"
    best_score = 0.0
    logging.info("Searching in knowledge base for candidates...")
    for identifier, values in tqdm.tqdm(kb.data.items()):
        for i, (index, context, candidate_embedding) in enumerate(
            zip(values["ENTITY_INDICES"], values["CONTEXTS"], values["EMBEDDINGS"])
        ):
            candidate = " ".join(context[i] for i in index)
            if mention.lower() == candidate.lower():
                if candidate_embedding is None:
                    candidate_embedding = _vectorize_context(
                        kb.pipeline, context, index
                    )
                    values["EMBEDDINGS"][i] = candidate_embedding
                score = _cosine_similarity(mention_embedding, candidate_embedding)
                if score > best_score:
                    best_score = score
                    best_candidate = identifier
    return best_candidate, best_score


def _vectorize_context(pipeline, context, index):
    index_mapping, features = extract_features(pipeline, context)
    aligned_indices = align_index(index, index_mapping)
    return pool_tokens(aligned_indices, features)


def _cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def _group_mentions(entities):
    mention = list()
    indices = list()
    original_indices = list()
    tokens = entities.reset_index().iterrows()
    for i, (j, token) in zip(entities.index, tokens):
        if token["entity"].startswith("B"):
            if mention:
                yield original_indices, indices, " ".join(mention)
                mention = list()
                indices = list()
                original_indices = list()
            indices.append(j)
            original_indices.append(i)
            mention.append(token["word"])
        elif token["entity"].startswith("I"):
            if indices[-1] == j - 1:
                indices.append(j)
                original_indices.append(i)
                mention.append(token["word"])
    if mention:
        yield original_indices, indices, " ".join(mention)
