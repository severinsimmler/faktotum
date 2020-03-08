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

    def rule_based(self):
        tp = 0
        fp = 0
        fn = 0

        for sentence in self.dataset:
            entity = dict()
            boundray = "[START]"
            ent = list()
            indices = list()
            last = "?"
            i_ = 9999999999999
            for i, token in enumerate(sentence):
                if token[1].startswith("B") and token[-1].startswith("Q"):
                    ent = [token[0]]
                    indices = [sentence.index(token)]
                    last = token[-1]
                elif (
                    token[1].startswith("I")
                    and token[-1].startswith("Q")
                    and i - 1 == i_
                ):
                    ent.append(token[0])
                    indices.append(i)
                    last = token[-1]
                else:
                    if ent:
                        text = re.sub(r'\s+([?.!"])', r"\1", " ".join(ent))
                        entity[text] = {"id": last, "indices": indices}
                i_ = i
            for text, identifier in entity.items():
                matches = defaultdict(list)
                success = False
                for key, value in self.kb.items():
                    if text in value["MENTIONS"]:
                        matches[text].append(key)
                        if identifier["id"] == key:
                            tp += 1
                            success = True
                            break
                        elif identifier["id"] != key:
                            fp += 1
                if len(matches[text]) == 0:
                    fn += 1
                    hard_to_disamiguate.append(
                        {
                            "mention": text,
                            "id": identifier["id"],
                            "index": identifier["indices"],
                            "sentence": sentence,
                            "candidates": [],
                        }
                    )
                elif len(matches[text]) > 1 and identifier["id"] in matches[text]:
                    hard_to_disamiguate.append(
                        {
                            "mention": text,
                            "id": identifier["id"],
                            "index": identifier["indices"],
                            "sentence": sentence,
                            "candidates": matches[text],
                        }
                    )
                elif len(matches[text]) > 1 and identifier["id"] not in matches[text]:
                    hard_to_disamiguate.append(
                        {
                            "mention": text,
                            "id": identifier["id"],
                            "index": identifier["indices"],
                            "sentence": sentence,
                            "candidates": [],
                        }
                    )
                elif not success and matches[text]:
                    hard_to_disamiguate.append(
                        {
                            "mention": text,
                            "id": identifier["id"],
                            "index": identifier["indices"],
                            "sentence": sentence,
                            "candidates": matches[text],
                        }
                    )

        precision = self.precision(tp, fp)
        recall = self.recall(tp, fn)
        return (
            {
                "precision": precision,
                "recall": recall,
                "f1": self.f1(precision, recall),
            },
            hard_to_disamiguate,
        )

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
