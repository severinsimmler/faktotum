import argparse
import csv
import collections
import json
import logging
from pathlib import Path
import time

import sklearn.manifold

import extract
from extract import exploration


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():
    logging.info("👋 Hi, you are about to cluster some documents.")

    parser = argparse.ArgumentParser(description="Filter sentences of a corpus.")
    parser.add_argument(
        "--topics", help="Path to the topics file produced by MALLET.", required=True
    )
    parser.add_argument(
        "--document-topics",
        help="Path to the document-topics file produced by MALLET.",
        required=True,
    )
    parser.add_argument(
        "--stopwords", help="Path to the stopwords list.", required=False
    )

    args = parser.parse_args()

    topics_filepath = Path(args.topics).resolve()
    document_topics_filepath = Path(args.document_topics).resolve()
    if args.stopwords:
        stopwords_filepath = Path(args.stopwords).resolve()
    else:
        stopwords_filepath = None

    model = exploration.TopicModel(
        topics_filepath, document_topics_filepath, stopwords_filepath
    )

    embedded = sklearn.manifold.TSNE(n_components=2, random_state=23, perplexity=15).fit_transform(
            model.document_topics.T.values
        )

    output = Path(
        topics_filepath.parent, f"{topics_filepath.stem}-{args.algorithm}.csv"
    )
    logging.info(f"Writing CSV file to {output.parent}...")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Dim1", "Dim2", "Topic"])
        for document, dominant_topic in zip(embedded, model.dominant_topics()):
            if args.stopwords:
                dominant_topic = ", ".join(model.topics.iloc[dominant_topic][:3])
            writer.writerow([document[0], document[1], dominant_topic])
