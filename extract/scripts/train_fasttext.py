import argparse
from pathlib import Path
import time

import extract


log = extract.logger(__file__)


def run():
    log.info("👋 Hi, you are about to train fastText on your corpus.")
    parser = argparse.ArgumentParser(description="Train fastText on a custom corpus.")
    parser.add_argument("--model", help="Path to the pre-trained fastText model.")
    parser.add_argument("--corpus", help="Path to the corpus directory.")
    parser.add_argument("--epochs", help="Number of epochs to train the model.")

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    model_path = Path(args.model)
    log.info(f"Corpus:\t{corpus_path.name}")
    log.info(f"Model:\t{model_path.name}")

    corpus = extract.Corpus(corpus_path)
    fasttext = extract.FastText(model_path)

    log.info("Starting training now...")
    fasttext.train(corpus, epochs=args.epochs)
    log.info("Training was successful!")

    now = int(time.time())
    model_path = Path(model_path.parent, f"{now}.fasttext")
    log.info(f"Saving model to {model_path}...")
    fasttext.model.save(model_path)
