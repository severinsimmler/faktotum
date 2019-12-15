import argparse
import json
import logging
from pathlib import Path

from flair.data import Sentence
from flair.models import SequenceTagger


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():

    parser = argparse.ArgumentParser(description="Filter sentences of a corpus.")
    parser.add_argument("--corpus", help="Path to the corpus JSON.", required=True)

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    logging.info("Loading tokenized corpus from file...")
    with corpus_path.open("r", encoding="utf-8") as corpus:
        corpus = json.load(corpus)

    tagger = SequenceTagger.load("de-ner")

    tagged_corpus = dict()
    for name, sentences in corpus.items():
        document = dict()
        for idx, sentence in sentences.items():
            text = " ".join(sentence)
            sentence = Sentence(text, use_tokenizer=False)
            tagger.predict(sentence)
            tagged_sentence = [
                {"text": t.text, "label": t.get_tag("ner").value, "belongs_to": None}
                for t in sentence
            ]
            for token in tagged_sentence:
                if token["label"] != "O":
                    if token["label"].split("-")[1] not in {"PER", "ORG"}:
                        token["label"] = "O"
            if not all(t["label"] == "O" for t in tagged_sentence):
                document[idx] = tagged_sentence
        if document:
            tagged_corpus[name] = document
            break

    with Path(f"{corpus_path.stem}-tagged.json").open("w", encoding="utf-8") as corpus:
        corpus.write(json.dumps(tagged_corpus, indent=2, ensure_ascii=False))
