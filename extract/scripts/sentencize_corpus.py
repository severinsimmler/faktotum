import argparse
import json
from pathlib import Path
import time

import extract


def run():
    log.info("👋 Hi, you are about to sentencize, tokenize and export a corpus.")

    parser = argparse.ArgumentParser(
        description="Sentencize, tokenize and export a text corpus as JSON."
    )
    parser.add_argument("--corpus", help="Path to the corpus directory.", required=True)

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    corpus = Corpus(corpus_path)

    log.info("Tokenizing corpus...")
    documents = json.dumps(extract.sentencize_corpus(corpus_path), ensure_ascii=False)

    export = Path(corpus_path.parent, f"{corpus_path.stem}.json")
    log.info(f"Exporting corpus to {export}...")
    export.write_text(documents, encoding="utf-8")
