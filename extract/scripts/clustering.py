import argparse
import json
import logging
from pathlib import Path
import time

import sklearn.decomposition

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

    model = exploration.TopicModel(topics_filepath, document_topics_filepath)

    pca = sklearn.decomposition.PCA(n_components=2, random_state=23)
    reduced = pca.fit(model.document_topics).transform(model.document_topics)
    reduced = np.append(reduced, [[topic] for topic in model.dominant_topics], axis=1)

    output = Path(topics_filepath.parent, "document-topics-pca.csv")
    np.savetxt(output, reduced, delimiter=",")
