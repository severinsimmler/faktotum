import argparse
import json
from pathlib import Path
import time

import extract
from extract import ontologia


log = extract.logger(__file__)


def run():
    log.info("👋 Hi, you are about to filter a corpus for annotation.")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--corpus", help="Path to the corpus JSON.", required=True)
    parser.add_argument("--words", help="", required=True)

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    words_path = Path(args.words).resolve()

    log.info("Loading words...")
    with words_path.open("r", encoding="utf-8") as words:
        words = json.load(words)

    log.info("Loading tokenized corpus from file...")
    with corpus_path.open("r", encoding="utf-8") as corpus:
        corpus = json.load(corpus)

    log.info(f"Filtering corpus with {len(words)} words...")
    filtered_corpus = json.dumps(
        dict(ontologia.filter_corpus(corpus, words)), ensure_ascii=False
    )

    output = Path(corpus_path.parent, f"{corpus_path.stem}-filtered.json")
    output.write_text(filtered_corpus, encoding="utf-8")
