"""
faktotum.pipelines
~~~~~~~~~~~~~~~~~~

This module implements high-level data pipeline functions.
"""

import logging

import numpy as np
import pandas as pd
import tqdm
from strsimpy.jaro_winkler import JaroWinkler

from faktotum.kb import KnowledgeBase
from faktotum.models import NamedEntityDisambiguation, NamedEntityRecognition
from faktotum.typing import Entities, Pipeline, TaggedTokens
from faktotum.utils import (
    align_index,
    cosine_similarity,
    extract_features,
    get_best_candidate,
    group_mentions,
    pool_tokens,
    predict_labels,
    sentencize,
    vectorize_context,
)

NER_MODELS = NamedEntityRecognition()
NED_MODELS = NamedEntityDisambiguation()
JARO_WINKLER = JaroWinkler()


def nel(text: str, kb: KnowledgeBase, domain: str) -> TaggedTokens:
    """Named Entity Linking.

    Parameters
    ----------
    text : str
        The text to process.
    kb : KnowledgeBase
        The knowledge base to link entities.
    domain : str
        Domain of the text, either `literary-texts` or `press-texts`.

    Returns
    -------
    TaggedTokens
        The tagged tokens.
    """
    tagged_tokens = ner(text, domain)
    return ned(tagged_tokens, kb, domain)


def ner(text: str, domain: str):
    """Named Entity Recognition.

    Parameters
    ----------
    text : str
        The text to process.
    domain : str
        Domain of the text, either `literary-texts` or `press-texts`.

    Returns
    -------
    TaggedTokens
        The tagged tokens.
    """
    pipeline = NER_MODELS[domain]
    sentences = [(i, sentence) for i, sentence in enumerate(sentencize(text))]
    predictions = list()
    logging.info("Processing sentences through NER pipeline...")
    for i, sentence in sentences:
        sentence = "".join(str(token) for token in sentence)
        prediction = predict_labels(pipeline, sentence, i)
        predictions.extend(prediction)
    return pd.DataFrame(predictions).loc[:, ["sentence_id", "word", "entity"]]


def ned(
    tokens: TaggedTokens,
    kb: KnowledgeBase,
    domain: str,
    candidate_threshold: float = 0.94,
):
    """Named Entity Disambiguation.

    Parameters
    ----------
    text : str
        The tagged tokens.
    domain : str
        Domain of the text, either `literary-texts` or `press-texts`.

    Returns
    -------
    TaggedTokens
        The tagged tokens.
    """
    pipeline = NED_MODELS[domain]
    identifiers = list()
    logging.info("Processing sentences through NED pipeline...")
    for sentence_id, sentence in tokens.groupby("sentence_id"):
        entities = sentence.dropna()
        index_mapping, features = extract_features(pipeline, sentence.loc[:, "word"])
        for original_index, index, mention in group_mentions(entities):
            aligned_index = align_index(index, index_mapping)
            mention_embedding = pool_tokens(aligned_index, features)
            best_candidate, score = get_best_candidate(
                mention, mention_embedding, kb, candidate_threshold
            )
            identifiers.append((original_index, best_candidate))
    tokens["entity_id"] = np.nan
    for mention, candidate in identifiers:
        tokens.iloc[mention, -1] = candidate
    return tokens
