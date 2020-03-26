import flair
import torch

flair.device = torch.device("cpu")

from pathlib import Path
from collections import defaultdict
import json
from collections import Counter
import re
import pandas as pd
import numpy as np
import tqdm
from flair.embeddings import BertEmbeddings
from flair.data import Sentence
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from faktotum.regression import Regression
import difflib
import random
from faktotum import utils
import statistics
from strsimpy.jaro_winkler import JaroWinkler
from faktotum import utils

random.seed(23)

JARO_WINKLER = JaroWinkler()
EMBEDDING = BertEmbeddings(
    "/mnt/data/users/simmler/model-zoo/entity-embeddings-smartdata-all-masked"
)


class EntityLinker:
    SIMILARITY_THRESHOLD = 0.942387

    def __init__(self, kb_dir: str):
        module_folder = Path(__file__).resolve().parent.parent
        self.corpus_folder = Path(module_folder, "data", "smartdata", "linking")
        self.train = list(self._load_corpus("train"))
        self.test = list(self._load_corpus("test"))
        self.dataset = self.train + self.test
        self.humans = json.loads(
            Path(kb_dir, "humans.json").read_text(encoding="utf-8")
        )
        self.organizations = json.loads(
            Path(kb_dir, "organizations.json").read_text(encoding="utf-8")
        )
        self.kb = self.humans.copy()
        self.kb.update(self.organizations.copy())

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
            if token[2].startswith("Q") and not current_entity:
                current_entity.append(token)
                current_id = token[2]
                last_index = i
            elif token[2].startswith("Q") and current_entity:
                if (
                    last_index + 1 == i
                    and current_id == token[2]
                    and token[1].startswith("I-")
                ):
                    current_entity.append(token)
                    last_index = i
                elif current_id != token[2]:
                    yield current_entity[0][2], current_entity
                    current_entity = [token]
                    last_index = i
                    current_id = token[2]
            elif not token[2].startswith("Q") and current_entity:
                yield current_entity[0][2], current_entity
                current_entity = list()
                last_index = i
                current_id = None
        if current_entity:
            yield current_entity[0][2], current_entity

    def rule_based(self):
        tp = 0
        fp = 0

        for sentence in tqdm.tqdm(self.test):
            spans = self.get_entity_spans(sentence)
            for identifier, entity in spans:
                text = " ".join([token[0] for token in entity])
                matches = set()
                if "-PER" in entity[0][2]:
                    kb = self.humans
                else:
                    kb = self.organizations
                for key, value in kb.items():
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
        return pd.Series({"precision": precision, "accuracy": accuracy,})

    @staticmethod
    def _vectorize(
        sentence,
        index,
        mask_entity: bool = False,
        return_type: bool = False,
        return_str: bool = False,
        return_id=False,
    ):
        for person, indices in index.items():
            tokens = list()
            entity = [token[0] for i, token in enumerate(sentence) if i in indices]
            type_ = "ORG" if any("ORG" in token[1] for token in sentence) else "PER"
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
            if return_id and return_str and return_type:
                yield person, type_, " ".join(entity), (vector / len(indices)).reshape(
                    1, -1
                )
            else:
                yield (vector / len(indices)).reshape(1, -1)

    @staticmethod
    def _string_similarity(a, b):
        return JARO_WINKLER.similarity(a, b)

    def _get_candidates(self, mention, is_org):
        candidates = set()
        mention = mention.lower()
        if is_org:
            kb = self.organizations
        else:
            kb = self.humans
        for key, value in kb.items():
            for context in value["MENTIONS"]:
                score = self._string_similarity(mention, context.lower())
                if score >= self.SIMILARITY_THRESHOLD:
                    candidates.add(key)
        return list(candidates)

    def similarities(self, mask_entity=False):
        tp = 0
        fp = 0
        tps = list()
        fps = list()
        num_candidates = list()
        for sentence in tqdm.tqdm(self.test):
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
                        sentence,
                        indices,
                        return_id=True,
                        return_type=True,
                        return_str=True,
                        mask_entity=mask_entity,
                    )
                )
                for identifier, type_, mention, mention_vector in mention_vectors:
                    max_score = 0.0
                    best_candidate = None
                    best_context = None
                    if type_ == "ORG":
                        is_org = True
                    else:
                        is_org = False
                    candidates = self._get_candidates(mention, is_org)
                    num_candidates.append(len(candidates))
                    print("Candidates:", len(candidates))
                    for candidate in candidates:
                        for context in self.kb[candidate]["MENTIONS"]:
                            if self.kb[candidate].get("DESCRIPTION"):
                                t = list(utils.tokenize(context))
                                t.extend(
                                    list(
                                        utils.tokenize(
                                            self.kb[candidate].get("DESCRIPTION")
                                        )
                                    )
                                )
                                text = " ".join(t)
                            else:
                                t = list(utils.tokenize(context))
                                text = " ".join(t)

                            indices = list(range(len(list(utils.tokenize(context)))))
                            sentence_ = Sentence(text, use_tokenizer=False)
                            EMBEDDING.embed(sentence_)
                            vector = sentence_[indices[0]].get_embedding().numpy()
                            for i in indices[1:]:
                                vector = vector + sentence_[i].get_embedding().numpy()
                            candidate_vector = (vector / len(indices)).reshape(1, -1)

                            score = cosine_similarity(mention_vector, candidate_vector)[
                                0
                            ][0]
                            if score > max_score:
                                max_score = score
                                best_candidate = candidate
                                best_context = context

                    if best_candidate == identifier:
                        tp += 1
                        tps.append({mention: best_context})
                    else:
                        fp += 1
                        fps.append({mention: best_context})
        with open("fps-tps.json", "w", encoding="utf-8") as f:
            json.dump({"tps": tps, "fps": fps}, f, ensure_ascii=False, indent=4)
        with open("scores.json", "w", encoding="utf-8") as f:
            json.dump({
            "accuracy": self.accuracy(tp, fp),
            "precision": self.precision(tp, fp),
            "num_candidates": statistics.mean(num_candidates),
            "embedding": "language-models/presse/multi",
        }, indent=4, ensure_ascii=False)
        return {
            "accuracy": self.accuracy(tp, fp),
            "precision": self.precision(tp, fp),
            "num_candidates": statistics.mean(num_candidates),
            "embedding": "language-models/presse/multi",
        }

    def _generate_data(self, data, mask_entity=False):
        X = list()
        y = list()
        for sentence in tqdm.tqdm(data):
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
                        sentence,
                        indices,
                        return_id=True,
                        return_type=True,
                        return_str=True,
                        mask_entity=mask_entity,
                    )
                )
                for identifier, type_, mention, mention_vector in mention_vectors:
                    true_candidate = self.kb.get(identifier)
                    if not true_candidate:
                        continue
                    else:
                        for context in true_candidate["MENTIONS"]:
                            t = list(utils.tokenize(context))
                            if mask_entity:
                                t = ["[MASK]" for _ in t]
                            if true_candidate.get("DESCRIPTION"):
                                t.extend(
                                    list(
                                        utils.tokenize(
                                            self.kb[identifier].get("DESCRIPTION")
                                        )
                                    )
                                )
                                text = " ".join(t)
                            else:
                                text = " ".join(t)
                            indices = list(range(len(list(utils.tokenize(context)))))
                            sentence_ = Sentence(text, use_tokenizer=False)
                            EMBEDDING.embed(sentence_)
                            vector = sentence_[indices[0]].get_embedding().numpy()
                            for i in indices[1:]:
                                vector = vector + sentence_[i].get_embedding().numpy()
                            candidate_vector = (vector / len(indices)).reshape(1, -1)
                            instance = np.concatenate(
                                (mention_vector[0], candidate_vector[0])
                            )
                            X.append(instance)
                            y.append(1.0)
                        if type_ == "ORG":
                            kb = self.organizations
                        else:
                            kb = self.humans
                        negative = random.sample(
                            [person for person in kb if person != identifier],
                            k=len(true_candidate["MENTIONS"]),
                        )
                        for id_ in negative:
                            negative_candidate = random.choice(kb[id_]["MENTIONS"])
                            t = list(utils.tokenize(negative_candidate))
                            if mask_entity:
                                t = ["[MASK]" for _ in t]
                            if kb[id_].get("DESCRIPTION"):
                                t.extend(
                                    list(
                                        utils.tokenize(self.kb[id_].get("DESCRIPTION"))
                                    )
                                )
                                text = " ".join(t)
                            else:
                                text = " ".join(t)
                            indices = list(
                                range(len(list(utils.tokenize(negative_candidate))))
                            )
                            sentence_ = Sentence(text, use_tokenizer=False)
                            EMBEDDING.embed(sentence_)
                            vector = sentence_[indices[0]].get_embedding().numpy()
                            for i in indices[1:]:
                                vector = vector + sentence_[i].get_embedding().numpy()
                            candidate_vector = (vector / len(indices)).reshape(1, -1)
                            instance = np.concatenate(
                                (mention_vector[0], candidate_vector[0])
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
        tp = 0
        fp = 0
        fn = 0
        for sentence in tqdm.tqdm(self.test):
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
                        sentence,
                        indices,
                        return_id=True,
                        return_type=True,
                        return_str=True,
                        mask_entity=mask_entity,
                    )
                )
                for identifier, type_, mention, mention_vector in mention_vectors:
                    max_score = 0.0
                    best_candidate = None
                    if type_ == "ORG":
                        is_org = True
                    else:
                        is_org = False
                    candidates = self._get_candidates(mention, is_org)
                    for candidate in candidates:
                        for context in self.kb[candidate]["MENTIONS"]:
                            t = list(utils.tokenize(context))
                            if mask_entity:
                                t = ["[MASK]" for _ in t]
                            if self.kb[candidate].get("DESCRIPTION"):
                                t.extend(
                                    list(
                                        utils.tokenize(
                                            self.kb[candidate].get("DESCRIPTION")
                                        )
                                    )
                                )
                                text = " ".join(t)
                            else:
                                text = " ".join(t)

                            indices = list(range(len(list(utils.tokenize(context)))))
                            sentence_ = Sentence(text, use_tokenizer=False)
                            EMBEDDING.embed(sentence_)
                            vector = sentence_[indices[0]].get_embedding().numpy()
                            for i in indices[1:]:
                                vector = vector + sentence_[i].get_embedding().numpy()
                            candidate_vector = (vector / len(indices)).reshape(1, -1)
                            instance = np.array(
                                np.concatenate((mention_vector[0], candidate_vector[0]))
                            ).reshape(1, -1)
                            score = model.predict(instance)[0]
                            if score > max_score:
                                max_score = score
                                best_candidate = candidate
                    if best_candidate == identifier:
                        tp += 1
                    else:
                        fp += 1
        result = {
            "accuracy": self.accuracy(tp, fp),
            "precision": self.precision(tp, fp),
        }
        return result

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
