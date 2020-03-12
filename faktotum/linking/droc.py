import flair
import torch

flair.device = torch.device("cpu")

from pathlib import Path
from collections import defaultdict
import json
import re
import pandas as pd
import tqdm
import numpy as np
from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from faktotum.regression import Regression
import random


EMBEDDING = BertEmbeddings("/mnt/data/users/simmler/model-zoo/entity-embeddings-droc")


class EntityLinker:
    def __init__(self):
        module_folder = Path(__file__).resolve().parent.parent
        self.corpus_folder = Path(module_folder, "data", "droc", "linking")
        self.train = self._load_corpus("train")
        test = self._load_corpus("test")
        dev = self._load_corpus("dev")
        self.dataset = dict()
        for i, data in enumerate([test, dev, self.train]):
            for key, value in data.items():
                self.dataset[f"{i}_{key}"] = value
        self.test = dict()
        for i, data in enumerate([test, dev]):
            for key, value in data.items():
                self.test[f"{i}_{key}"] = value

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

    def _load_corpus(self, dataset: str):
        textfile = Path(self.corpus_folder, f"{dataset}.txt")
        with textfile.open("r", encoding="utf-8") as file_:
            return json.load(file_)

    def _build_knowledge_base(
        self,
        novel,
        threshold: int = 1,
        mask_entity: bool = False,
        build_embeddings=True,
    ):
        context = defaultdict(list)
        mentions = defaultdict(set)
        embeddings = defaultdict(list)
        for sentence in tqdm.tqdm(novel):
            for i, token in enumerate(sentence):
                if token[2] != "-":
                    if sentence not in context[token[2]]:
                        context[token[2]].append(sentence)
                        if build_embeddings:
                            vector = next(
                                self._vectorize(
                                    sentence,
                                    index={token[2]: [i]},
                                    mask_entity=mask_entity,
                                )
                            )
                            embeddings[token[2]].append(vector)
        for sentence in novel:
            for token in sentence:
                if token[2] != "-":
                    if token[0] not in mentions[token[2]]:
                        mentions[token[2]].add(token[0])

        kb = defaultdict(dict)
        for key in mentions:
            if len(context[key]) > threshold:
                kb[key]["CONTEXTS"] = context[key]
                if build_embeddings:
                    kb[key]["EMBEDDINGS"] = embeddings[key]
                kb[key]["MENTIONS"] = mentions[key]
        return {key: value for key, value in kb.items() if value}

    @staticmethod
    def _vectorize(sentence, index, mask_entity: bool = False, return_id=False):
        for person, indices in index.items():
            tokens = list()
            for i, token in enumerate(sentence):
                if i in indices and mask_entity:
                    tokens.append("[MASK]")
                else:
                    tokens.append(token[0])
            text = " ".join(tokens)
            sentence_ = Sentence(text, use_tokenizer=False)
            EMBEDDING.embed(sentence_)
            vector = sentence_[indices[0]].get_embedding().numpy()
            for i in indices[1:]:
                vector = vector + sentence_[i].get_embedding().numpy()
            if return_id:
                yield person, (vector / len(indices)).reshape(1, -1)
            else:
                yield (vector / len(indices)).reshape(1, -1)

    def similarities(self, mask_entity=False):
        stats = list()
        for novel in tqdm.tqdm(self.test.values()):
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
                    mention_vectors = list(
                        self._vectorize(
                            sentence, indices, return_id=True, mask_entity=mask_entity
                        )
                    )

                    for identifier, mention_vector in mention_vectors:
                        max_score = 0.0
                        best_candidate = None
                        for person, contexts in kb.items():
                            for context, candidate_vector in zip(
                                contexts["CONTEXTS"], contexts["EMBEDDINGS"]
                            ):
                                if context != sentence:
                                    score = cosine_similarity(
                                        mention_vector, candidate_vector
                                    )
                                    if score > max_score:
                                        max_score = score
                                        best_candidate = person

                        if best_candidate == identifier:
                            tp += 1
                        else:
                            fp += 1

            stats.append(
                {"accuracy": self.accuracy(tp, fp), "precision": self.precision(tp, fp)}
            )
        return pd.DataFrame(stats).describe()

    def rule_based(self):
        stats = list()
        for novel in tqdm.tqdm(self.test.values()):
            tp = 0
            fp = 0
            fn = 0
            kb = self._build_knowledge_base(novel, build_embeddings=False)
            for sentence in novel:
                mentions = [token for token in sentence if token[2] != "-"]
                for mention in mentions:
                    matches = set()
                    for values in kb.values():
                        if len(values["CONTEXTS"]) == 1:
                            skip = True
                            continue
                        skip = False
                        valid_sentences = list()
                        for context in values["CONTEXTS"]:
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
                            fp += 1
                        elif len(matches) == 1:
                            if list(matches)[0] == mention[2]:
                                tp += 1
                            else:
                                fp += 1
                        else:
                            # If ambiguous, it's a FN
                            fp += 1
            stats.append(
                {"accuracy": self.accuracy(tp, fp), "precision": self.precision(tp, fp)}
            )
        return pd.DataFrame(stats).describe()

    def _generate_data(self, dataset, mask_entity=True):
        X = list()
        y = list()
        data = list()
        for novel in dataset.values():
            data.extend(novel)
        kb = self._build_knowledge_base(
            data, build_embeddings=True, threshold=2, mask_entity=mask_entity
        )
        for sentence in data:
            is_mentioned = [token for token in sentence if token[2] != "-"]
            if not is_mentioned:
                continue
            if is_mentioned:
                indices = defaultdict(list)
                for i, token in enumerate(sentence):
                    if token[2] != "-":
                        indices[token[2]].append(i)
                mention_vectors = list(
                    self._vectorize(
                        sentence, indices, return_id=True, mask_entity=mask_entity
                    )
                )

                for identifier, mention_vector in mention_vectors:
                    if identifier in kb:
                        candidates = kb[identifier]["EMBEDDINGS"]
                        for candidate in candidates:
                            instance = np.concatenate((mention_vector[0], candidate[0]))
                            X.append(instance)
                            y.append(1.0)
                        negative = random.sample(
                            [person for person in kb if person != identifier],
                            k=len(candidates),
                        )
                        for id_ in negative:
                            negative_candidate = random.choice(kb[id_]["EMBEDDINGS"])
                            instance = np.concatenate(
                                (mention_vector[0], negative_candidate[0])
                            )
                            X.append(instance)
                            y.append(0.0)
        return np.array(X), np.array(y)

    def regression(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        generate_data=False,
        mask_entity=True,
    ):
        if generate_data:
            X_train, y_train = self._generate_data(self.train)
            X_test, y_test = self._generate_data(self.test)
        model = Regression()
        history = model.fit(X_train, y_train)
        test_mse_score, test_mae_score = model.evaluate(X_test, y_test)
        print("MSE", test_mse_score)
        print("MAE", test_mae_score)

        # EVALUATION
        stats = list()
        for novel in tqdm.tqdm(self.test.values()):
            tp = 0
            fp = 0
            fn = 0
            kb = self._build_knowledge_base(
                novel, mask_entity=mask_entity, build_embeddings=True
            )
            for sentence in novel:
                is_mentioned = [token for token in sentence if token[2] != "-"]
                if not is_mentioned:
                    continue
                elif is_mentioned:
                    indices = defaultdict(list)
                    for i, token in enumerate(sentence):
                        if token[2] != "-":
                            indices[token[2]].append(i)
                    mention_vectors = list(
                        self._vectorize(
                            sentence, indices, return_id=True, mask_entity=mask_entity
                        )
                    )

                    for identifier, mention_vector in mention_vectors:
                        max_score = 0.0
                        best_candidate = None
                        for person, contexts in kb.items():
                            for context, candidate_vector in zip(
                                contexts["CONTEXTS"], contexts["EMBEDDINGS"]
                            ):
                                if context != sentence:
                                    instance = np.array(
                                        np.concatenate(
                                            (mention_vector[0], candidate_vector[0])
                                        )
                                    )
                                    score = model.predict(instance)[0]
                                    print(score)
                                    if score > max_score:
                                        max_score = score
                                        best_candidate = person

                        if best_candidate == identifier:
                            tp += 1
                        else:
                            fp += 1
            result = {
                "accuracy": self.accuracy(tp, fp),
                "precision": self.precision(tp, fp),
            }
            print(result)
            stats.append(result)
        return pd.DataFrame(stats).describe()
