import argparse
import collections
import csv
import json
import logging
import time
from pathlib import Path

import sklearn.decomposition

import extract
from extract import exploration

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():
    logging.info("👋 Hi, you are about to decomposite some documents.")

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
        "--stopwords", help="Path to the stopwords list.", required=True
    )

    args = parser.parse_args()

    topics_filepath = Path(args.topics).resolve()
    document_topics_filepath = Path(args.document_topics).resolve()
    stopwords_filepath = Path(args.stopwords).resolve()

    model = exploration.TopicModel(
        topics_filepath, document_topics_filepath, stopwords_filepath
    )

    pca = sklearn.decomposition.PCA(n_components=2, random_state=23)
    reduced = pca.fit(model.document_topics).transform(model.document_topics.values)
    pc1, pc2 = pca.explained_variance_ratio_
    logging.info(f"PC1: {round(pc1 * 100)}%")
    logging.info(f"PC2: {round(pc2 * 100)}%")

    output = Path(topics_filepath.parent, f"{topics_filepath.stem}-pca.csv")
    logging.info(f"Writing CSV file to {output.parent}...")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Dim1", "Dim2", "Topic"])
        for document, dominant_topic in zip(reduced, model.dominant_topics()):
            words = ", ".join(model.topics.iloc[dominant_topic][:3])
            writer.writerow([document[0], document[1], words])
