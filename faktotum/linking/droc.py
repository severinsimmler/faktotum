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
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING = BertEmbeddings("/mnt/data/users/simmler/model-zoo/ner-droc")


class EntityLinker:
    def __init__(self):
        module_folder = Path(__file__).resolve().parent.parent
        self.corpus_folder = Path(module_folder, "data", "droc", "linking")
        self.train = self._load_corpus("train")
        self.test = self._load_corpus("test")
        self.dev = self._load_corpus("dev")
        self.dataset = dict()
        for i, data in enumerate([self.test, self.dev, self.train]):
            for key, value in data.items():
                self.dataset[f"{i}_{key}"] = value

    def _load_corpus(self, dataset: str):
        textfile = Path(self.corpus_folder, f"{dataset}.txt")
        with textfile.open("r", encoding="utf-8") as file_:
            return json.load(file_)

    def _build_knowledge_base(self, novel, threshold: int = 1, mask_entity: bool = False):
        context = defaultdict(list)
        mentions = defaultdict(set)
        embeddings = defaultdict(list)
        for sentence in novel:
            for i, token in enumerate(sentence):
                if token[2] != "-":
                    if sentence not in context[token[2]]:
                        context[token[2]].append(sentence)
                        vector = next(self._vectorize(sentence, index={token[2]: [i]}, mask_entity=False))
                        embeddings[token[2]].append(vector)
        for sentence in novel:
            for token in sentence:
                if token[2] != "-":
                    if token[0] not in mentions[token[2]]:
                        mentions[token[2]].add(token[0])

        kb = defaultdict(dict)
        for key in mentions:
            if len(context[key]) > threshold:
                kb[key]["CONTEXT"] = context[key]
                kb[key]["EMBEDDINGS"] = embeddings[key]
                kb[key]["MENTIONS"] = mentions[key]
        return kb

    @staticmethod
    def _vectorize(sentence, index, mask_entity: bool = False):
        for person, indices in index.items():
            tokens = list()
            for i, token in enumerate(sentence):
                if i in indices and mask_entity:
                    tokens.append("[MASK]")
                else:
                    tokens.append(token[0])
            text = " ".join(tokens)
            sentence = Sentence(text, use_tokenizer=False)
            EMBEDDING.embed(sentence)
            vector = sentence[indices[0]].get_embedding().numpy()
            for i in indices[1:]:
                vector = vector + sentence[i].get_embedding().numpy()
            yield (vector / len(indices)).reshape(1, -1)

    def similarities(self, mask_entity=False):
        stats = list()
        for novel in tqdm.tqdm(self.dataset.values()):
            tp = 0
            fp = 0
            fn = 0
            kb = self._build_knowledge_base(novel)
            for sentence in novel:
                is_mentioned = [token for token in sentence if token[2] != "-"]
                if not is_mentioned:
                    continue
                if is_mentioned:
                    indices = defaultdict(list)
                    for i, token in enumerate(sentence):
                        if token[2] != "-":
                            indices[token[2]].append(i)
                    mention_vectors = list(self._vectorize(sentence, indices))
                    for identifier, mention_vector in mention_vectors:
                        for person, contexts in kb.items():
                            for context in contexts["CONTEXT"]:
                                if context != sentence:
                                    candidate = defaultdict(list)
                                    for i, token in enumerate(context):
                                        if token[2] == person:
                                            indices[person].append(i)
                                    candidate_vector = list(self._vectorize(context, candidate))[0][1]
                                    print(mention_vector, candidate_vector)
                                    print(candidate_vector)
                                    print(cosine_similarity(mention_vector, candidate_vector))
                        return


                for mention, mention_vector in zip(mentions, mention_vectors):
                    matches = defaultdict(list)
                    for values in kb.values():
                        if len(values["CONTEXT"]) == 1:
                            skip = True
                            continue
                        skip = False
                        valid_sentences = list()
                        for context in values["CONTEXT"]:
                            # Filter the current sentence
                            if context != sentence:
                                valid_sentences.append(context)
                        for context in valid_sentences:
                            for i, token in enumerate(context):
                                if token[2] != "-" and token[0] == mention[0]:
                                    vector = list(
                                            self._vectorize(
                                                context,
                                                index={token[2]: [i]},
                                                mask_entity=mask_entity,
                                            )
                                        )
                                    matches[token[2]].append(vector)
                    if not skip:
                        if len(matches) == 0:
                            fn += 1
                        elif len(matches) == 1:
                            if list(matches)[0] == mention[2]:
                                tp += 1
                            else:
                                fp += 1
                        else:
                            max_score = 0.0
                            candidate = None
                            for identifier, vector in matches.items():
                                score = cosine_similarity(mention_vector[0], vector[0][0])[0][0]
                                if score > max_score:
                                    max_score = score
                                    candidate = identifier
                            if candidate == mention[2]:
                                tp += 1
                            else:
                                fp += 1
            try:
                precision = self.precision(tp, fp)
                recall = self.recall(tp, fn)
                f1 = self.f1(precision, recall)
                stats.append(
                    {"precision": precision, "recall": recall, "f1": f1,}
                )
            except ZeroDivisionError:
                pass
        return pd.DataFrame(stats).describe()

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
                        if len(values["CONTEXT"]) == 1:
                            skip = True
                            continue
                        skip = False
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
                    if not skip:
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
            try:
                precision = self.precision(tp, fp)
                recall = self.recall(tp, fn)
                f1 = self.f1(precision, recall)
                stats.append(
                    {"precision": precision, "recall": recall, "f1": f1,}
                )
            except ZeroDivisionError:
                pass
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
