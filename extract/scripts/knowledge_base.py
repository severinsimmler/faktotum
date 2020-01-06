import argparse
import json
import logging

import extract


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def run():
    logging.info("👋 Hi, enriching your knowledge base with DBpedia.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", required=True)

    args = parser.parse_args()

    kb = extract.knowledge.enrich_knowledge_base(args.kb)

    output = Path(Path(args.kb).parent, "kb-enriched.json")
    with output.open("r", encoding="utf-8") as file_:
        file_.write(json.dumps(kb, indent=2, ensure_ascii=False))
