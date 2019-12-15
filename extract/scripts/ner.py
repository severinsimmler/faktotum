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
    progress = len(corpus)
    for i, (name, sentences) in enumerate(corpus.items()):
        logging.info(f"{progress - i} documents remaining...")
        document = dict()
        for idx, sentence in sentences.items():
            text = " ".join(sentence)
            sentence = Sentence(text, use_tokenizer=False)
            tagger.predict(sentence)
            tagged_sentence = [
                {"text": t.text, "label": t.get_tag("ner").value, "belongs_to": None}
                for t in sentence
            ]
            for span in sentence.get_spans("ner"):
                span_ = " ".join([token.text for token in span.tokens])
                if span_ not in entities:
                    entities[span_] = {
                        "id": len(entities),
                        "class": [span.tag],
                        "occurences": {name: [idx]},
                    }
                else:
                    if span.tag not in entities[span_]["class"]:
                        entities[span_]["class"].append(span.tag)
                    if name not in entities[span_]["occurences"]:
                        entities[span_]["occurences"][name] = [idx]
                    elif name in entities[span_]["occurences"]:
                        if idx not in entities[span_]["occurences"][name]:
                            entities[span_]["occurences"][name].append(idx)
                indices = [token.idx - 1 for token in span.tokens]
                for index in indices:
                    tagged_sentence[index]["belongs_to"] = entities[span_]["id"]

            if not all(t["label"] == "O" for t in tagged_sentence):
                document[idx] = tagged_sentence

        if document:
            tagged_corpus[name] = document

            with Path(corpus_path.parent, f"{corpus_path.stem}-tagged.json").open("w", encoding="utf-8") as corpus:
                corpus.write(json.dumps(tagged_corpus, indent=2, ensure_ascii=False))

            with Path(corpus_path.parent, f"{corpus_path.stem}-knowledge-base.json").open(
                "w", encoding="utf-8"
            ) as knowledge:
                knowledge.write(json.dumps(entities, indent=2, ensure_ascii=False))
