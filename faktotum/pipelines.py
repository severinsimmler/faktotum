import transformers
import pandas as pd
import faktotum
import tqdm
from faktotum.typing import Entities, Pipeline, KnowledgeBase, TaggedTokens
import numpy as np

MODELS = {"ner": {"literary-texts": "severinsimmler/literary-german-bert", "press-texts": "severinsimmler/german-press-bert"},
          "ned": {"literary-texts": "severinsimmler/literary-german-bert", "press-texts": "severinsimmler/bert-base-german-press-cased"}}



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
    for token_id, vector in zip(pipeline.tokenizer.encode(sentence), np.squeeze(pipeline(sentence))):
        token = pipeline.tokenizer.decode([token_id])
        if token not in {"[CLS]", "[SEP]", "[MASK]"}:
            if token["word"].startswith("##"):
                vectors[-1] += vector
            else:
                vectors.append(vector)
    return vectors


def nel(text: str, kb: KnowledgeBase, domain: str = "literary-texts"):
    tagged_tokens = ner(text, domain)
    return ned(tagged_tokens, kb, domain)


def ner(text: str, domain: str = "literary-texts"):
    model_name = MODELS["ner"][domain]
    pipeline = transformers.pipeline("ner", model=model_name, tokenizer=model_name, ignore_labels=[])
    sentences = [(i, sentence) for i, sentence in enumerate(faktotum.sentencize(text))]
    predictions = list()
    for i, sentence in tqdm.tqdm(sentences):
        sentence = "".join(str(token) for token in sentence)
        prediction = _predict_labels(pipeline, sentence, i)
        predictions.extend(prediction)
    return pd.DataFrame(predictions).loc[:, ["sentence_id", "word", "entity", "score"]]


def ned(tokens: TaggedTokens, kb: KnowledgeBase = None, domain: str = "literary-texts"):
    model_name = MODELS["ned"][domain]
    pipeline = transformers.pipeline("feature-extraction", model=model_name, tokenizer=model_name)
    for sentence_id, sentence in tokens.groupby("sentence_id"):
        sentence = sentence.reset_index(drop=True)
        text = " ".join(sentence["word"])
        entity_indices = sentence[sentence["entity"] != "O"]
        features = _extract_features(pipeline, text)
        yield features