from pathlib import Path
from collections import defaultdict
from collections import defaultdict
import json


class EntityLinker:
    def __init__(self, corpus: str, kb_dir: str):
        module_folder = Path(__file__).resolve().parent.parent
        self.corpus = corpus
        self.corpus_folder = Path(module_folder, "data", corpus)
        self.train = list(self._load_corpus("train"))
        self.test = list(self._load_corpus("test"))
        self.dev = list(self._load_corpus("dev"))
        self.dataset = self.train + self.test + self.dev
        if self.corpus == "droc":
            self.kb = self._build_knowledge_base()
        else:
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

    def _build_knowledge_base(self):
        context = defaultdict(list)
        mentions = defaultdict(set)
        for sentence in self.dataset:
            for token in sentence:
                if token[2] != "-":
                    if sentence not in context[token[2]]:
                        context[token[2]].append(sentence)
        for sentence in self.dataset:
            for token in sentence:
                if token[2] != "-":
                    if token[0] not in mentions[token[2]]:
                        mentions[token[2]].add(token[0])

        kb = defaultdict(dict)
        for key in mentions:
            kb[key]["context"] = context[key]
            kb[key]["mentions"] = mentions[key]
        return kb

    def rule_based(self):
        tp = 0
        fp = 0
        fn = 0
        hard_to_disamiguate = list()
        for key, value in self.kb.items():
            if self.corpus == "droc":
                # ignore entities occuring only once
                if len(value["context"]) > 1:
                    for sentence in value["context"]:
                        mentions = defaultdict(set)
                        for sentence_ in value["context"]:
                            if sentence != sentence_:
                                for token in sentence_:
                                    if token[2] != "-":
                                        mentions[token[2]].add(token[0])
                        for token in sentence:
                            if token[2] != "-":
                                for key, mention in mentions.items():
                                    if token[2] == key and token[0] in mention:
                                        tp += 1
                                        break
                                    elif token[2] == key and token[0] not in mention:
                                        fn += 1
                                        if sentence not in hard_to_disamiguate:
                                            hard_to_disamiguate.append(
                                                {
                                                    "mention": token[0],
                                                    "id": token[2],
                                                    "index": sentence.index(token),
                                                    "sentence": sentence,
                                                }
                                            )
                                    elif token[2] != key and token[0] in mention:
                                        fp += 1
            elif corpus == "smartdata":
                for sentence in self.dataset:
                    entity = defaultdict(list)
                    for token in sentence:
                        if token[-1].startswith("Q"):
                            entity[token[-1]].append(token[0])

                    for identifier, tokens in entity.items():
                        text = " ".join(tokens)
                        matches = defaultdict(list)
                        for key, value in self.kb.items():
                            if text in value["MENTIONS"]:
                                matches[text].append(key)
                                if identifier == key:
                                    tp += 1
                                elif identifier != key:
                                    fp += 1
                        if len(matches[text]) == 0:
                            fn += 1
                            print(text)

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
