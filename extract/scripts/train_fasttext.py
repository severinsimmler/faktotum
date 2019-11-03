import argparse
from pathlib import Path
import time

import extract


def run():
    log.info("👋 Hi, you are about to train a fastText model.")

    parser = argparse.ArgumentParser(description="Train fastText on a custom corpus.")
    parser.add_argument("--model", help="Path to pre-trained model.", required=False)
    parser.add_argument("--corpus", help="Path to the corpus directory.", required=True)
    parser.add_argument("--window", help="Sliding window.", required=False, type=int)
    parser.add_argument("--epochs", help="Epochs to train.", required=True, type=int)

    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    if args.model:
        model_path = Path(args.model).resolve()
        if model_path.suffix == ".bin":
            mode = "facebook-pretrained"
        else:
            mode = "custom-pretrained"
    else:
        model_path = None
        mode = "plain"

    corpus = extract.Corpus(corpus_path)
    fasttext = extract.FastText(pretrained_model=model_path, window=args.window)

    log.info("Tokenizing corpus...")
    tokens = list(corpus.tokens())
    log.info(f"Training with {len(tokens)} documents.")
    fasttext.train(tokens, epochs=args.epochs)

    model_path = str(Path(corpus_path.parent, f"{corpus_path.stem}-{mode}.fasttext"))
    log.info(f"Saving model to {model_path}...")
    fasttext.model.save(model_path)
