import argparse
import json
from pathlib import Path
import time

import extract


def run():
    log.info("👋 Hi, you are about to tokenize and export a corpus.")

    parser = argparse.ArgumentParser(
        description="Tokenize a corpus and export as JSON."
    )
    parser.add_argument("--corpus", help="Path to the corpus directory.", required=True)

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    corpus = extract.Corpus(corpus_path)

    log.info("Tokenizing corpus...")
    documents = json.dumps(
        {
            name: list(extract.sentencize(text, tokenize=True))
            for name, text in corpus.documents(yield_name=True)
        },
        ensure_ascii=False,
    )

    export = Path(corpus_path.parent, f"{corpus_path.stem}.json")
    log.info(f"Exporting corpus to {export}...")
    export.write_text(documents, encoding="utf-8")
