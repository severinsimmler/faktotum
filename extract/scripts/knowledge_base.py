import argparse
import json
from pathlib import Path
import logging

import extract


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def load_dump(filepath):
    with open(filepath, "r", encoding="utf-8") as file_:
        for line in file_:
            yield json.loads(line)


def run():
    logging.info("👋 Hi, filtering entities from a knowledge base.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", required=True)

    args = parser.parse_args()

    kwargs = {
        "humans": load_dump(args.kb) if "human" in Path(args.kb).name else None,
        "organizations": load_dump(args.kb)
        if "business" in Path(args.kb).name
        else None,
        "positions": load_dump(args.kb) if "position" in Path(args.kb).name else None,
    }

    kb = extract.knowledge.KnowledgeBase(**kwargs)
    kb.export(".")
