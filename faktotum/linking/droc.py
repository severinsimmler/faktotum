from pathlib import Path
from collections import defaultdict
from collections import defaultdict


class EntityLinker:
    def __init__(self):
        module_folder = Path(__file__).resolve().parent.parent
        self.corpus_folder = Path(module_folder, "data", "droc")

        self.train = list(self._load_corpus("train"))
        self.test = list(self._load_corpus("test"))
        self.dev = list(self._load_corpus("dev"))
        self.dataset = self.train + self.test + self.dev
        self.kb = self._build_knowledge_base()

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
