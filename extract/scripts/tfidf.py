import argparse
import logging
import time
from pathlib import Path

import extract

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():
    logging.info(
        "👋 Hi, you are about to calculate a TF-IDF weighted co-occurence matrix."
    )

    parser = argparse.ArgumentParser(description="Train fastText on a custom corpus.")
    parser.add_argument("--corpus", help="Path to the corpus directory.", required=True)

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    corpus = extract.load_corpus(corpus_path)

    tfidf = extract.TfIdf()
    tfidf.fit_weights(corpus)

    corpus = extract.load_corpus(corpus_path)
    tfidf.build_matrix(corpus)

    output = Path(corpus_path.parent, corpus_path.stem)
    logging.info(f"Saving model to {output.parent}...")
    tfidf.save(output)
