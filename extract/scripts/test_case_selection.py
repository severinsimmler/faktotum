import argparse
import json
import logging
from pathlib import Path

from extract.exploration import calculate_sentence_similarities, select_new_sentences


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Path to the corpus JSON.", required=True)
    parser.add_argument("--model", help="Path or name to BERT model.", required=True)
    parser.add_argument(
        "--reference-sentences",
        help="Path to file with reference sentences to compare with.",
        required=True,
    )
    parser.add_argument("--n", help="Sample size.", type=int, default=5000)

    args = parser.parse_args()

    ref, matrix = calculate_sentence_similarities(args.corpus, args.model, args.reference_sentences, args.n)
    matrix.to_csv("similarities.csv")
    new_sentences = list(set(select_new_sentences(matrix, ref)))
    with Path("new-sentences.json").open("w", encoding="utf-8") as file_:
        file_.write(json.dumps(new_sentences, indent=2))
