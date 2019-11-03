import argparse
import json
import logging
from pathlib import Path
import time

import extract
from extract import ontologia


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():
    logging.info("👋 Hi, you are about to filter sentences of a corpus.")

    parser = argparse.ArgumentParser(description="Filter sentences of a corpus.")
    parser.add_argument("--corpus", help="Path to the corpus JSON.", required=True)
    parser.add_argument(
        "--words", help="Key words which must occur in a sentence", required=True
    )

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    words_path = Path(args.words).resolve()

    logging.info("Loading words...")
    with words_path.open("r", encoding="utf-8") as words:
        words = json.load(words)

    logging.info("Loading tokenized corpus from file...")
    with corpus_path.open("r", encoding="utf-8") as corpus:
        corpus = json.load(corpus)

    logging.info(f"Filtering corpus with {len(words)} words...")
    filtered_corpus = json.dumps(dict(ontologia.filter_corpus(corpus, words)))

    output = Path(corpus_path.parent, f"{corpus_path.stem}-filtered.json")
    output.write_text(filtered_corpus, encoding="utf-8")
