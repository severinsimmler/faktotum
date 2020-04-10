import numpy as np
import pandas as pd
import tqdm
import transformers

from faktotum.utils import (
    sentencize,
    MODELS,
    pool_tokens,
    extract_features,
    align_index,
)
from faktotum.kb import KnowledgeBase
from faktotum.typing import Entities, Pipeline, TaggedTokens
from strsimpy.jaro_winkler import JaroWinkler
import logging


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


JARO_WINKLER = JaroWinkler()


def nel(text: str, kb: KnowledgeBase, similarity_threshold=0.94, domain: str = "literary-texts"):
    tagged_tokens = ner(text, domain)
    return ned(tagged_tokens, kb, similarity_threshold, domain)


def ner(text: str, domain: str = "literary-texts"):
    model_name = MODELS["ner"][domain]
    logging.info("Loading named entity recognition model...")
    pipeline = transformers.pipeline(
        "ner", model=model_name, tokenizer=model_name, ignore_labels=[]
    )
    sentences = [(i, sentence) for i, sentence in enumerate(sentencize(text))]
    predictions = list()
    logging.info("Start processing sentences through NER pipeline...")
    for i, sentence in tqdm.tqdm(sentences):
        sentence = "".join(str(token) for token in sentence)
        prediction = _predict_labels(pipeline, sentence, i)
        predictions.extend(prediction)
    return pd.DataFrame(predictions).loc[:, ["sentence_id", "word", "entity"]]


def ned(
    tokens: TaggedTokens,
    kb: KnowledgeBase = None,
    similarity_threshold: float = 0.94,
    domain: str = "literary-texts",
):
    model_name = MODELS["ned"][domain]
    logging.info("Loading feature extraction model...")
    pipeline = transformers.pipeline(
        "feature-extraction", model=model_name, tokenizer=model_name
    )
    identifiers = list()
    logging.info("Start processing sentences through NEL pipeline...")
    for sentence_id, sentence in tqdm.tqdm(tokens.groupby("sentence_id")):
        entities = sentence.dropna()
        index_mapping, features = extract_features(pipeline, sentence.loc[:, "word"])
        for original_index, index, mention in _group_mentions(entities):
            aligned_index = align_index(index, index_mapping)
            mention_embedding = pool_tokens(aligned_index, features)
            best_candidate, score = _get_best_candidate(
                mention, mention_embedding, kb, similarity_threshold
            )
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


def _get_best_candidate(mention, mention_embedding, kb, similarity_threshold):
    best_candidate = "NIL"
    best_score = 0.0
    logging.info("Searching in knowledge base for candidates...")
    for identifier, values in tqdm.tqdm(kb.items()):
        for i, (index, context, candidate_embedding) in enumerate(
            zip(values["ENTITY_INDICES"], values["CONTEXTS"], values["EMBEDDINGS"])
        ):
            candidate = " ".join(context[i] for i in index)
            if mention.lower() in candidate.lower() or JARO_WINKLER.similarity(mention, candidate) >= similarity_threshold:
                if not candidate_embedding:
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
            if mention[-1] == j - 1:
                indices.append(j)
                original_indices.append(i)
                mention.append(token["word"])
    if mention:
        yield original_indices, indices, " ".join(mention)
