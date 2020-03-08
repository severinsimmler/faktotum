from pathlib import Path
from collections import defaultdict
import json
import re
import tqdm


class EntityLinker:
    def __init__(self, kb_dir: str):
        module_folder = Path(__file__).resolve().parent.parent
        self.corpus_folder = Path(module_folder, "data", "smartdata")
        self.train = list(self._load_corpus("train"))
        self.test = list(self._load_corpus("test"))
        self.dev = list(self._load_corpus("dev"))
        self.dataset = self.train + self.test + self.dev
        h = Path(kb_dir, "humans.json").read_text(encoding="utf-8")
        o = Path(kb_dir, "organizations.json").read_text(encoding="utf-8")
        self.kb = json.loads(h)
        self.kb.update(json.loads(o))

    def _load_corpus(self, dataset: str):
        sentence = list()
        text = Path(self.corpus_folder, f"{dataset}.txt").read_text(encoding="utf-8")
        for line in text.split("\n"):
            if not line.startswith("#"):
                if line != "":
                    sentence.append(line.split(" "))
                else:
                    yield sentence
                    sentence = list()
        if sentence:
            yield sentence

    @staticmethod
    def get_entity_spans(sentence):
        current_entity = list()
        last_index = -1
        current_id = None
        for i, token in enumerate(sentence):
            if (
                token[2].startswith("Q")
                and last_index + 1 == i
                and (current_id is not None or token[2] == current_id)
            ):
                current_entity.append(token)
            elif token[2].startswith("Q") and last_index + 1 != i:
                if current_entity:
                    yield current_id, current_entity
                current_entity = [token]
            current_id = token[2]
            last_index = i
        if current_entity:
            yield current_id, current_entity

    def rule_based(self):
        tp = 0
        fp = 0

        for sentence in self.dataset:
            spans = self.get_entity_spans(sentence)
            for identifier, entity in spans:
                text = " ".join([token[0] for token in entity])
                matches = set()
                for key, value in self.kb.items():
                    if text in value["MENTIONS"]:
                        matches.add(key)
                if len(matches) < 1:
                    fp += 1
                elif len(matches) == 1:
                    if list(matches)[0] == identifier:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fp += 1

        precision = self.precision(tp, fp)
        accuracy = self.accuracy(tp, fp)
        return {
            "precision": precision,
            "accuracy": accuracy,
        }

    @staticmethod
    def precision(tp: int, fp: int) -> float:
        return tp / (tp + fp)

    @staticmethod
    def recall(tp: int, fn: int) -> float:
        return tp / (tp + fn)

    @staticmethod
    def f1(precision: float, recall: float) -> float:
        return 2 * ((precision * recall) / (precision + recall))

    @staticmethod
    def accuracy(tp: int, fp: int) -> float:
        return tp / (tp + fp)
