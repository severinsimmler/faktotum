import transformers
import pandas as pd
import faktotum
import tqdm


MODELS = {"ner": {"literary-texts": "severinsimmler/literary-german-bert", "press-texts": "severinsimmler/german-press-bert"},
          "ned": {"literary-texts": "severinsimmler/literary-german-bert", "press-texts": "severinsimmler/bert-base-german-press-cased"}}


def _predict(pipeline, text, sentence_id):
    entities = list()
    for token in pipeline(text):
        token["sentence_id"] = sentence_id
        if token["word"].startswith("##"):
            entities[-1]["word"] += token["word"][2:]
        else:
            entities.append(token)
    return entities


def ner(text: str, domain: str = "literary-texts"):
    model_name = MODELS["ner"][domain]
    pipeline = transformers.pipeline("ner", model=model_name, tokenizer=model_name, ignore_labels=[])
    sentences = list(faktotum.sentencize(text))
    predictions = list()
    for i, sentence in tqdm.tqdm(enumerate(sentences)):
        prediction = _predict(pipeline, sentence, sentence_id)
        predictions.append(prediction)
    return pd.DataFrame(predictions).loc[:, ["sentence_id", "word", "entity", "score"]]
