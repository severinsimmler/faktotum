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
import difflib
from faktotum import utils

EMBEDDING = BertEmbeddings("/mnt/data/users/simmler/model-zoo/ner-droc")


class EntityLinker:
    _string_similarity_threshold = .9

    def __init__(self, kb_dir: str):
        module_folder = Path(__file__).resolve().parent.parent
        self.corpus_folder = Path(module_folder, "data", "smartdata")
        self.train = list(self._load_corpus("train"))
        self.test = list(self._load_corpus("test"))
        self.dev = list(self._load_corpus("dev"))
        self.dataset = self.train + self.test + self.dev
        self.humans = json.loads(Path(kb_dir, "humans.json").read_text(encoding="utf-8"))
        self.organizations = json.loads(Path(kb_dir, "organizations.json").read_text(encoding="utf-8"))
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
            if (
                token[2].startswith("Q")
                and last_index + 1 == i
                and (current_id is not None or token[2] == current_id)
            ):
                current_entity.append(token)
            elif token[2].startswith("Q") and last_index + 1 != i:
                if current_entity:
                    yield current_entity[0][2], current_entity
                current_entity = [token]
            current_id = token[2]
            last_index = i
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
    def _vectorize(sentence, index, mask_entity: bool = False, return_type: bool = False, return_str: bool = False, return_id=False):
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
                yield person, type_, " ".join(entity), (vector / len(indices)).reshape(1, -1)
            else:
                yield (vector / len(indices)).reshape(1, -1)

    @staticmethod
    def _string_similarity(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _get_candidates(self, mention, is_org):
        candidates = set()
        mention = mention.lower()
        if is_org:
            kb = self.organizations
        else:
            kb = self.humans
        for key, value in tqdm.tqdm(kb.items()):
            for context in value["MENTIONS"]:
                score = self._string_similarity(mention, context.lower())
                if score >= self._string_similarity_threshold:
                    candidates.add(key)
        return candidates

    def similarities(self, mask_entity=False):
        tp = 0
        fp = 0
        for sentence in tqdm.tqdm(self.dataset):
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
                        sentence, indices, return_id=True, return_type=True, return_str=True, mask_entity=mask_entity
                    )
                )
                for identifier, type_, mention, mention_vector in mention_vectors:
                    max_score = 0.0
                    best_candidate = None
                    if type_ == "ORG":
                        is_org = True
                    else:
                        is_org = False
                    for candidate in self._get_candidates(mention, is_org):
                        for context in self.kb[candidate]["MENTIONS"]:
                            if self.kb[candidate].get("DESCRIPTION"):
                                text = context + " " + self.kb[candidate].get("DESCRIPTION")
                            else:
                                text = context
                            
                            indices = list(range(len(list(utils.tokenize(context)))))
                            sentence_ = Sentence(text, use_tokenizer=False)
                            EMBEDDING.embed(sentence_)
                            vector = sentence_[indices[0]].get_embedding().numpy()
                            for i in indices[1:]:
                                vector = vector + sentence_[i].get_embedding().numpy()
                            candidate_vector = (vector / len(indices)).reshape(1, -1)

                            score = cosine_similarity(mention_vector, candidate_vector)
                            print(score)
                            if score > max_score:
                                max_score = score
                                best_candidate = person

                    if best_candidate == identifier:
                        tp += 1
                    else:
                        fp += 1

        return {"accuracy": self.accuracy(tp, fp), "precision": self.precision(tp, fp)}


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
