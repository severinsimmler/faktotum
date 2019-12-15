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
    entities = dict()
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
            if not all(t["label"] == "O" for t in tagged_sentence):
                document[idx] = tagged_sentence
            for span in sentence.get_spans("ner"):
                span = " ".join([token.text for token in span.tokens])
                if span not in entities:
                    entities[span] = {"id": len(entities), "class": span.tag.value}
        if document:
            tagged_corpus[name] = document
            break

    print(entities)
    with Path(f"{corpus_path.stem}-tagged.json").open("w", encoding="utf-8") as corpus:
        corpus.write(json.dumps(tagged_corpus, indent=2, ensure_ascii=False))
