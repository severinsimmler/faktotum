import flair
import torch
flair.device = torch.device("cpu")

from pathlib import Path
from collections import defaultdict
import json
import re
import pandas as pd
import tqdm
from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from sklearn.cluster import AgglomerativeClustering

EMBEDDING = BertEmbeddings("/mnt/data/users/simmler/model-zoo/ner-droc")


class EntityLinker:
    def __init__(self):
        module_folder = Path(__file__).resolve().parent.parent
        self.corpus_folder = Path(module_folder, "data", "droc", "linking")
        self.train = self._load_corpus("train")
        self.test = self._load_corpus("test")
        self.dev = self._load_corpus("dev")
        self.dataset = dict()
        self.dataset.update(self.train)
        self.dataset.update(self.dev)
        self.dataset.update(self.test)

    def _load_corpus(self, dataset: str):
        textfile = Path(self.corpus_folder, f"{dataset}.txt")
        with textfile.open("r", encoding="utf-8") as file_:
            return json.load(file_)

    @staticmethod
    def _build_knowledge_base(novel, threshold: int = 1):
        context = defaultdict(list)
        mentions = defaultdict(set)
        for sentence in novel:
            for token in sentence:
                if token[2] != "-":
                    if sentence not in context[token[2]]:
                        context[token[2]].append(sentence)
        for sentence in novel:
            for token in sentence:
                if token[2] != "-":
                    if token[0] not in mentions[token[2]]:
                        mentions[token[2]].add(token[0])
        kb = defaultdict(dict)
        for key in mentions:
            if len(context[key]) >= threshold:
                kb[key]["CONTEXT"] = context[key]
                kb[key]["MENTIONS"] = mentions[key]
        return kb

    @staticmethod
    def _vectorize(sentence, mask_entity: bool = False):
        tokens = list()
        index = defaultdict(list)
        for i, token in enumerate(sentence):
            if token[2] == "-":
                tokens.append(token[0])
            else:
                if index[token[2]]:
                    if index[token[2]][-1] + 1 == i:
                        index[token[2]].append(i)
                    else:
                        print("ERROR: Same entity multiple times.")
                else:
                    index[token[2]].append(i)
                if mask_entity:
                    tokens.append("[MASK]")
                else:
                    tokens.append(token[0])
        text = " ".join(tokens)
        sentence = Sentence(text, use_tokenizer=False)
        EMBEDDING.embed(sentence)
        for entity, indices in index.items():
            vector = None
            for i in indices:
                if not vector:
                    vector = sentence[i].get_embedding().numpy()
                else:
                    vector = vector + sentence[i].get_embedding().numpy()
            yield entity, len(indices)

    def clustering(self):
        for novel in tqdm.tqdm(self.dataset.values()):
            kb = self._build_knowledge_base(novel)
            matrix = np.array([self._vectorize(sentence) for sentence in novel])
            clusters = AgglomerativeClustering().fit_transform(matrix)

    def rule_based(self):
        stats = list()
        for novel in tqdm.tqdm(self.dataset.values()):
            tp = 0
            fp = 0
            fn = 0
            kb = self._build_knowledge_base(novel)
            for sentence in novel:
                mentions = [token for token in sentence if token[2] != "-"]
                for mention in mentions:
                    matches = set()
                    for values in kb.values():
                        valid_sentences = list()
                        for context in values["CONTEXT"]:
                            # Filter the current sentence
                            if context != sentence:
                                valid_sentences.extend(context)
                        mentions_ = [
                            token for token in valid_sentences if token[2] != "-"
                        ]
                        for mention_ in mentions_:
                            if mention[0] == mention_[0]:
                                matches.add(mention_[2])
                    if len(matches) == 0:
                        fn += 1
                    elif len(matches) == 1:
                        if list(matches)[0] == mention[2]:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        # If ambiguous, it's a FN
                        fn += 1
            precision = self.precision(tp, fp)
            recall = self.recall(tp, fn)
            f1 = self.f1(precision, recall)
            stats.append(
                {"precision": precision, "recall": recall, "f1": f1,}
            )
        return pd.DataFrame(stats).describe()

    @staticmethod
    def precision(tp: int, fp: int) -> float:
        return tp / (tp + fp)

    @staticmethod
    def recall(tp: int, fn: int) -> float:
        return tp / (tp + fn)

    @staticmethod
    def f1(precision: float, recall: float) -> float:
        return 2 * ((precision * recall) / (precision + recall))
