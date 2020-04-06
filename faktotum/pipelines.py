import numpy as np
import pandas as pd
import tqdm
import sklearn.metrics.pairwise
import transformers

import faktotum
from faktotum.typing import Entities, KnowledgeBase, Pipeline, TaggedTokens

MODELS = {
    "ner": {
        "literary-texts": "severinsimmler/literary-german-bert",
        "press-texts": "severinsimmler/german-press-bert",
    },
    "ned": {
        "literary-texts": "severinsimmler/literary-german-bert",
        "press-texts": "severinsimmler/bert-base-german-press-cased",
    },
}


def nel(text: str, kb: KnowledgeBase, domain: str = "literary-texts"):
    tagged_tokens = ner(text, domain)
    return ned(tagged_tokens, kb, domain)


def ner(text: str, domain: str = "literary-texts"):
    model_name = MODELS["ner"][domain]
    pipeline = transformers.pipeline(
        "ner", model=model_name, tokenizer=model_name, ignore_labels=[]
    )
    sentences = [(i, sentence) for i, sentence in enumerate(faktotum.sentencize(text))]
    predictions = list()
    for i, sentence in tqdm.tqdm(sentences):
        sentence = "".join(str(token) for token in sentence)
        prediction = _predict_labels(pipeline, sentence, i)
        predictions.extend(prediction)
    return pd.DataFrame(predictions).loc[:, ["sentence_id", "word", "entity", "score"]]


def ned(tokens: TaggedTokens, kb: KnowledgeBase = None, domain: str = "literary-texts"):
    model_name = MODELS["ned"][domain]
    pipeline = transformers.pipeline(
        "feature-extraction", model=model_name, tokenizer=model_name
    )
    for sentence_id, sentence in tokens.groupby("sentence_id"):
        sentence.loc[:, "entity_id"] = "O"
        text = " ".join(sentence.loc[:, "word"])
        entities = sentence[sentence.loc[:, "entity"] != "O"]
        mentions = _group_mentions(entities)
        features = _extract_features(pipeline, text)
        for mention in mentions:
            vector = _pool_entity(mention, features)
            best_candidate, score = _get_best_candidate(vector, kb)
            print(mention, best_candidate, score)
            sentence.iloc[mention, -1] = best_candidate
        print(sentence)
    return tokens


class KnowledgeBase:
    def __init__(self, data, domain):
        self.data = data
        self._model_name = MODELS["ned"][domain]
        self._pipeline = transformers.pipeline("feature-extraction", model=self._model_name, tokenizer=self._model_name)
        self._vectorize_contexts()

    def _vectorize_contexts(self):
        for key, value in self.data.items():
            for indices, context in zip(value["ENTITY_INDICES"], value["CONTEXTS"]):
                features = _extract_features(self._pipeline, context)
                embeddings = _pool_entity(indices, features)
                self.data[key]["EMBEDDINGS"] = embeddings

    def items(self):
        for key, value in self.data.items():
            yield key, value


def _predict_labels(pipeline: Pipeline, sentence: str, sentence_id: int) -> Entities:
    entities = list()
    for token in pipeline(sentence):
        if token["word"] not in {"[CLS]", "[SEP]", "[MASK]"}:
            token["sentence_id"] = sentence_id
            if token["word"].startswith("##"):
                entities[-1]["word"] += token["word"][2:]
            else:
                entities.append(token)
    return entities


def _extract_features(pipeline: Pipeline, sentence: str) -> Entities:
    vectors = list()
    for token_id, vector in zip(
        pipeline.tokenizer.encode(sentence), np.squeeze(pipeline(sentence))
    ):
        token = pipeline.tokenizer.decode([token_id])
        if token not in {"[CLS]", "[SEP]", "[MASK]"} and not token.startswith("##"):
            vectors.append(vector)
    return vectors


def _get_best_candidate(vector, kb):
    best_candidate = "NIL"
    best_score = 0.0
    for identifier, values in kb.items():
        for candidate in values["EMBEDDINGS"]:
            vector = vector.reshape(1, -1)
            candidate = candidate.reshape(1, -1)
            print(vector.shape)
            print(candidate.shape)
            score = sklearn.metrics.pairwise.cosine_similarity(vector, candidate)
            if score > best_score:
                best_score = score
                best_candidate = identifier
    return best_candidate, best_score


def _pool_entity(indices, features):
    entity = features[indices[0]]
    for index in indices[1:]:
        entity += features[index]
    return entity


def _group_mentions(entities):
    mention = list()
    for row, token in entities.iterrows():
        if token["entity"].startswith("B"):
            if mention:
                yield mention
                mention = list()
            mention.append(row)
        elif token["entity"].startswith("I"):
            if mention:
                if mention[-1] == row - 1:
                    mention.append(row)
    if mention:
        yield mention
