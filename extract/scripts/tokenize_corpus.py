import argparse
import json
import logging
from pathlib import Path
import time

import extract


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():
    logging.info("👋 Hi, you are about to sentencize, tokenize and export a corpus.")

    parser = argparse.ArgumentParser(
        description="Sentencize, tokenize and export a text corpus as JSON."
    )
    parser.add_argument("--corpus", help="Path to the corpus directory.", required=True)

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()

    logging.info("Tokenizing corpus...")
    documents = json.dumps(extract.sentencize_corpus(corpus_path))

    export = Path(corpus_path.parent, f"{corpus_path.stem}.json")
    logging.info(f"Exporting corpus to {export}...")
    export.write_text(documents, encoding="utf-8")
