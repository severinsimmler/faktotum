import argparse
import logging
from pathlib import Path
import time

import extract


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():
    logging.info("👋 Hi, you are about to train a word2vec model.")

    parser = argparse.ArgumentParser(description="Train word2vec on a custom corpus.")
    parser.add_argument(
        "--model", help="Path to pre-trained model (optional).", required=False
    )
    parser.add_argument("--corpus", help="Path to the corpus directory.", required=True)
    parser.add_argument(
        "--algorithm",
        help="Algorithm to use, either 'cbow' or 'skipgram'.",
        default="cbow",
    )
    parser.add_argument("--epochs", help="Epochs to train.", required=True, type=int)

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    if args.model:
        model_path = Path(args.model).resolve()
        word2vec = extract.Word2Vec.load(model_path)
        mode = "pretrained-cbow"
    else:
        sg = {"skipgram": 1, "cbow": 0}.get(args.algorithm, 0)
        word2vec = extract.Word2Vec(sg=sg)
        mode = f"plain-{args.algorithm}"

    corpus = extract.load_corpus(corpus_path)
    word2vec.train(corpus, epochs=args.epochs)

    model_path = Path(corpus_path.parent, f"{corpus_path.stem}-{mode}.word2vec")
    logging.info(f"Saving model to {model_path}...")
    word2vec.save(model_path)
