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
from faktotum.research.regression import Regression
import difflib
import random
from faktotum import utils
import statistics
from strsimpy.jaro_winkler import JaroWinkler
from faktotum import utils
from faktotum.research.similarity import EntitySimilarityLearner, EntityEmbeddings
from strsimpy.jaro_winkler import JaroWinkler

random.seed(23)

JARO_WINKLER = JaroWinkler()
# model = EntitySimilarityLearner.load(
#    "/mnt/data/users/simmler/model-zoo/similarity-lstm-smartdata/best-model.pt"
# )
EMBEDDING = BertEmbeddings(
    "/mnt/data/users/simmler/model-zoo/bert-multi-presse-adapted"
)  # model.source_embeddings


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
        prediction = list()
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
                    prediction.append({"pred": "NIL", "gold": identifier})
                elif len(matches) == 1:
                    prediction.append({"pred": list(matches)[0], "gold": identifier})
                    if list(matches)[0] == identifier:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fp += 1
        precision = self.precision(tp, fp)
        accuracy = self.accuracy(tp, fp)
        with open("prediction.json", "w", encoding="utf-8") as f:
            json.dump(prediction, f)
        return pd.Series({"precision": precision, "accuracy": accuracy,})

    def string_similarities(self):
        tp = 0
        fp = 0
        prediction = list()
        for sentence in tqdm.tqdm(self.test):
            spans = self.get_entity_spans(sentence)
            for identifier, entity in spans:
                max_score = 0.0
                best_identifier = None
                text = " ".join([token[0] for token in entity])
                if "-PER" in entity[0][2]:
                    kb = self.humans
                else:
                    kb = self.organizations
                for key, value in kb.items():
                    for mention in value["MENTIONS"]:
                        score = self._string_similarity(mention, text)
                        if score > max_score:
                            max_score = score
                            best_identifier = key
                prediction.append({"pred": best_identifier, "gold": identifier})
                if identifier == best_identifier:
                    tp += 1
                else:
                    fp += 1
        precision = self.precision(tp, fp)
        accuracy = self.accuracy(tp, fp)
        with open("prediction.json", "w", encoding="utf-8") as f:
            json.dump(prediction, f)
        return pd.Series({"precision": precision, "accuracy": accuracy,})

    @staticmethod
    def get_persons(sent):
        persons = dict()

        for token in sent:
            if token[2] != "-":
                persons[token[2]] = list()

        for i, token in enumerate(sent):
            if token[2] != "-":
                if persons[token[2]] and persons[token[2]][-1][-1] == i - 1:
                    # wenn entität aus mehreren tokens besteht
                    persons[token[2]][-1].append(i)
                elif persons[token[2]] and persons[token[2]][-1] != i - 1:
                    # wenn entität mehrmals im satz vorkommt
                    persons[token[2]].append([i])
                elif not persons[token[2]]:
                    # wenn entität zum ersten mal vorkommt
                    persons[token[2]].append([i])
        return persons

    @staticmethod
    def _vectorize(
        sentence,
        persons,
        mask_entity: bool = False,
        return_type: bool = False,
        return_str: bool = False,
        return_id=False,
    ):
        for person, indices in persons.items():
            tokens = list()
            type_ = "ORG" if any("ORG" in token[1] for token in sentence) else "PER"
            for i, token in enumerate(sentence):
                if i in indices and mask_entity:
                    tokens.append("[MASK]")
                else:
                    tokens.append(token[0])
            text = " ".join(tokens)
            sentence_ = Sentence(text, use_tokenizer=False)
            if isinstance(EMBEDDING, EntityEmbeddings):
                for mention in indices:
                    entity = [
                        token[0] for i, token in enumerate(sentence) if i in mention
                    ]
                    EMBEDDING.embed(sentence_, [mention])
                    if return_id and return_str and return_type:
                        yield person, type_, " ".join(
                            entity
                        ), sentence_.embedding.detach().numpy().reshape(1, -1)
                    else:
                        yield sentence_.embedding.detach().numpy().reshape(1, -1)
            else:
                EMBEDDING.embed(sentence_)
                for mention in indices:
                    entity = [
                        token[0] for i, token in enumerate(sentence) if i in mention
                    ]
                    vector = sentence_[mention[0]].get_embedding().numpy()
                    for i in mention[1:]:
                        vector = vector + sentence_[i].get_embedding().numpy()
                    if return_id and return_str and return_type:
                        yield person, type_, " ".join(entity), (
                            vector / len(mention)
                        ).reshape(1, -1)
                    else:
                        yield (vector / len(mention)).reshape(1, -1)

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
            if " " + mention + " " in " ".join(value["MENTIONS"]).lower():
                candidates.add(key)
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
        prediction = list()
        num_candidates = list()
        for sentence in tqdm.tqdm(self.test):
            is_mentioned = [token for token in sentence if token[2] != "-"]
            if not is_mentioned:
                continue
            if is_mentioned:
                persons = self.get_persons(sentence)
                mention_vectors = list(
                    self._vectorize(
                        sentence,
                        persons,
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
                    best_sent = None
                    if type_ == "ORG":
                        is_org = True
                    else:
                        is_org = False
                    candidates = self._get_candidates(mention, is_org)
                    num_candidates.append(len(candidates))
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
                            if isinstance(EMBEDDING, EntityEmbeddings):
                                EMBEDDING.embed(sentence_, [indices])
                                candidate_vector = (
                                    sentence_.embedding.detach().numpy().reshape(1, -1)
                                )
                            else:
                                EMBEDDING.embed(sentence_)
                                vector = sentence_[indices[0]].get_embedding().numpy()
                                for i in indices[1:]:
                                    vector = (
                                        vector + sentence_[i].get_embedding().numpy()
                                    )
                                candidate_vector = (vector / len(indices)).reshape(
                                    1, -1
                                )

                            score = cosine_similarity(mention_vector, candidate_vector)[
                                0
                            ][0]
                            if score > max_score:
                                max_score = score
                                best_candidate = candidate
                                best_context = context
                                best_sent = text

                    prediction.append({"pred": best_candidate, "gold": identifier})
                    if best_candidate == identifier:
                        tp += 1
                        tps.append(
                            {
                                "true": mention,
                                "pred": best_context,
                                "true_id": identifier,
                                "pred_id": best_candidate,
                                "score": float(max_score),
                                "sentence": " ".join([token[0] for token in sentence]),
                                "context": " ".join([token[0] for token in best_sent]),
                            }
                        )
                    else:
                        fp += 1
                        if best_sent:
                            fps.append(
                                {
                                    "true": mention,
                                    "pred": best_context,
                                    "true_id": identifier,
                                    "pred_id": best_candidate,
                                    "score": float(max_score),
                                    "sentence": " ".join(
                                        [token[0] for token in sentence]
                                    ),
                                    "context": " ".join(
                                        [token[0] for token in best_sent]
                                    ),
                                }
                            )
        with open("fps-tps.json", "w", encoding="utf-8") as f:
            json.dump({"tps": tps, "fps": fps}, f, ensure_ascii=False, indent=4)
        with open("scores.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": self.accuracy(tp, fp),
                    "precision": self.precision(tp, fp),
                    "num_candidates": statistics.mean(num_candidates),
                    "embedding": "language-models/presse/multi",
                },
                f,
                indent=4,
                ensure_ascii=False,
            )
        with open("prediction.json", "w", encoding="utf-8") as f:
            json.dump(prediction, f)
        return {
            "accuracy": self.accuracy(tp, fp),
            "precision": self.precision(tp, fp),
            "num_candidates": statistics.mean(num_candidates),
            "embedding": "language-models/presse/multi",
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
