import argparse
import csv
import collections
import json
import logging
from pathlib import Path
import time

import sklearn.cluster

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

    args = parser.parse_args()

    topics_filepath = Path(args.topics).resolve()
    document_topics_filepath = Path(args.document_topics).resolve()

    logging.info("Loading topic model data...")
    model = exploration.TopicModel(
        topics_filepath, document_topics_filepath
    )

    logging.info("Clustering with k-Means...")
    k = sklearn.cluster.KMeans(n_clusters=10, n_jobs=-1, random_state=23)
    k.fit(model.document_topics.T.values)
    clusters = k.labels_

    output = Path(topics_filepath.parent, f"{topics_filepath.stem}-clusters.csv")
    logging.info(f"Writing CSV file to {output.parent}...")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Dim1", "Dim2", "Cluster"])
        for document, cluster in zip(embedded, clusters):
            writer.writerow([document[0], document[1], cluster])

    output = Path(topics_filepath.parent, f"{topics_filepath.stem}-centers.csv")
    logging.info(f"Writing JSON file to {output.parent}...")
    with output.open("w", encoding="utf-8") as f:
        f.write(json.dumps(k.cluster_centers_.tolist(), indent=2))
