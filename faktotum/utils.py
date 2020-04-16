"""
faktotum.utils
~~~~~~~~~~~~~

This module implements general helper functions.
"""

from typing import Generator, List

import numpy as np
import syntok.segmenter
import syntok.tokenizer

from faktotum.typing import Entities, KnowledgeBase, Pipeline, TaggedTokens

TOKENIZER = syntok.tokenizer.Tokenizer()


def tokenize(text: str) -> Generator[str, None, None]:
    """Split text into tokens.

    Parameters
    ----------
    text
        The text to split into tokens.

    Yields
    ------
    One token at a time.
    """
    for token in TOKENIZER.tokenize(text):
        yield str(token).strip()


def sentencize(text: str) -> Generator[str, None, None]:
    """Split text into sentences.

    Parameters
    ----------
    text
        The text to split into tokenized sentences.

    Yields
    ------
    One sentence at a time.
    """
    for paragraph in syntok.segmenter.process(text):
        for sentence in paragraph:
            yield sentence


def pool_tokens(indices, features):
    entity = features[indices[0]]
    for index in indices[1:]:
        entity += features[index]
    return entity


def extract_features(pipeline: Pipeline, sentence: List[str]) -> Entities:
    vectors = list()
    text = " ".join(sentence)
    index = dict()
    for i, token in enumerate(sentence):
        subtokens = [
            t for t in pipeline.tokenizer.tokenize(token) if not t.startswith("##")
        ]
        index[i] = subtokens
    for token, vector in zip(
        pipeline.tokenizer.tokenize(text), np.squeeze(pipeline(text))
    ):
        if token not in {"[CLS]", "[SEP]", "[MASK]"} and not token.startswith("##"):
            vectors.append(vector)
    return index, vectors


def align_index(original_indices, subtoken_indices):
    aligned_indices = list()
    for i in original_indices:
        for j, subtokens in enumerate(subtoken_indices[i]):
            aligned_indices.append(i + j)
    return aligned_indices


def predict_labels(pipeline: Pipeline, sentence: str, sentence_id: int) -> Entities:
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


def get_best_candidate(mention, mention_embedding, kb, candidate_threshold):
    best_candidate = "NIL"
    best_score = 0.0
    logging.info("Searching in knowledge base for candidates...")
    for identifier, values in tqdm.tqdm(kb.data.items()):
        for i, (index, context, candidate_embedding) in enumerate(
            zip(values["ENTITY_INDICES"], values["CONTEXTS"], values["EMBEDDINGS"])
        ):
            candidate = " ".join(context[i] for i in index)
            if (
                JARO_WINKLER.similarity(mention.lower(), candidate.lower())
                >= candidate_threshold
            ):
                if candidate_embedding is None:
                    candidate_embedding = vectorize_context(kb.pipeline, context, index)
                    values["EMBEDDINGS"][i] = candidate_embedding
                score = cosine_similarity(mention_embedding, candidate_embedding)
                if score > best_score:
                    best_score = score
                    best_candidate = identifier
    return best_candidate, best_score


def vectorize_context(pipeline, context, index):
    index_mapping, features = extract_features(pipeline, context)
    aligned_indices = align_index(index, index_mapping)
    return pool_tokens(aligned_indices, features)


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def group_mentions(entities):
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
