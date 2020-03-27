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
from faktotum.similarity import EntitySimilarity
import random
import torch

#EMBEDDING = BertEmbeddings("/mnt/data/users/simmler/model-zoo/ner-droc")


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
        #self.test = dict()
        #for i, data in enumerate([test, dev]):
        #    for key, value in data.items():
        #        self.test[f"{i}_{key}"] = value

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
        similarity_model=False, source=False, target=True
    ):
        context = defaultdict(list)
        mentions = defaultdict(list)
        embeddings = defaultdict(list)
        for sentence in tqdm.tqdm(novel):
            for i, token in enumerate(sentence):
                if token[2] != "-":
                    if sentence not in context[token[2]]:
                        mentions[token[2]].append(token[0])
                        context[token[2]].append(sentence)
                        if build_embeddings:
                            vector = next(
                                self._vectorize(
                                    sentence,
                                    persons={token[2]: [i]},
                                    mask_entity=mask_entity,
                                    similarity_model=similarity_model,
                                    source=source,
                                    target=target
                                )
                            )
                            embeddings[token[2]].append(vector)
        kb = defaultdict(dict)
        for key in mentions:
            if len(context[key]) > threshold:
                kb[key]["CONTEXTS"] = context[key]
                if build_embeddings:
                    kb[key]["EMBEDDINGS"] = embeddings[key]
                kb[key]["MENTIONS"] = mentions[key]
        return {key: value for key, value in kb.items() if value}

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


    def _vectorize(self, sentence, persons, mask_entity: bool = False, return_id=False, return_str=False, similarity_model=False, source=False, target=False):
        for person, indices in persons.items():
            for mention in indices:
                tokens = list()
                for i, token in enumerate(sentence):
                    if i in indices and mask_entity:
                        tokens.append("[MASK]")
                    else:
                        tokens.append(token[0])
                text = " ".join(tokens)
                sentence_ = Sentence(text, use_tokenizer=False)
                if not similarity_model:
                    EMBEDDING.embed(sentence_)
                else:
                    if source:
                        e = self.network.source_embedding
                    elif target:
                        e = self.network.target_embedding
                    e.embed(sentence_)
                vector = sentence_[indices[0]].get_embedding().numpy()
                name = [sentence_[indices[0]].text]
                for i in indices[1:]:
                    vector = vector + sentence_[i].get_embedding().numpy()
                    name.append(sentence_[i].text)
                if return_id:
                    if return_str:
                        yield person, (vector / len(indices)).reshape(1, -1), " ".join(name)
                    else:
                        yield person, (vector / len(indices)).reshape(1, -1)
                else:
                    if return_str:
                        yield (vector / len(indices)).reshape(1, -1), " ".join(name)
                    else:
                        yield (vector / len(indices)).reshape(1, -1)



    def similarities(self, mask_entity=False):
        stats = list()
        for i, novel in enumerate(self.test.values()):
            tp = 0
            fp = 0
            fn = 0
            tps = list()
            fps = list()
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
                            sentence, indices, return_id=True, mask_entity=mask_entity, return_str=True
                        )
                    )

                    for identifier, mention_vector, name in mention_vectors:
                        max_score = 0.0
                        best_candidate = None
                        best_mention = None
                        best_sent = None
                        for person, contexts in kb.items():
                            for context, candidate_vector, mention in zip(
                                contexts["CONTEXTS"], contexts["EMBEDDINGS"], contexts["MENTIONS"]
                            ):
                                if context != sentence:
                                    score = cosine_similarity(
                                        mention_vector, candidate_vector
                                    )
                                    if score > max_score:
                                        max_score = score
                                        best_candidate = person
                                        best_mention = mention
                                        best_sent = context

                        if best_candidate == identifier:
                            tp += 1
                            tps.append({"true": name,
                                        "pred": best_mention,
                                        "true_id": identifier,
                                        "pred_id": best_candidate,
                                        "score": float(max_score[0][0]),
                                        "sentence": " ".join([token[0] for token in sentence]),
                                        "context": " ".join([token[0] for token in best_sent])})
                        else:
                            fp += 1
                            fps.append({"true": name,
                                        "pred": best_mention,
                                        "true_id": identifier,
                                        "pred_id": best_candidate,
                                        "score": float(max_score[0][0]),
                                        "sentence": " ".join([token[0] for token in sentence]),
                                        "context": " ".join([token[0] for token in best_sent])})
            with open(f"droc-{i}.json", "w", encoding="utf-8") as f:
                json.dump({"tps": tps, "fps": fps}, f, ensure_ascii=False, indent=4)
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
        return pd.DataFrame(stats)

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

    def model(self, model_path, mask_entity=False):
        self.network = EntitySimilarity.load(model_path)
        stats = list()
        for i, novel in enumerate(self.test.values()):
            tp = 0
            fp = 0
            fn = 0
            tps = list()
            fps = list()
            kb = self._build_knowledge_base(novel, similarity_model=True, source=False, target=True)
            for sentence in novel:
                is_mentioned = [token for token in sentence if token[2] != "-"]
                if not is_mentioned:
                    continue
                if is_mentioned:
                    persons = self.get_persons(sentence)
                    
                    mention_vectors = list(
                        self._vectorize(
                            sentence, persons, return_id=True, mask_entity=mask_entity, return_str=True, similarity_model=True, source=True, target=False
                        )
                    )

                    for identifier, mention_vector, name in mention_vectors:
                        max_score = 0.0
                        best_candidate = None
                        best_mention = None
                        best_sent = None
                        for person, contexts in kb.items():
                            for context, candidate_vector, mention in zip(
                                contexts["CONTEXTS"], contexts["EMBEDDINGS"], contexts["MENTIONS"]
                            ):
                                if context != sentence:
                                    score = network.get_similarity(
                                        torch.tensor(mention_vector), torch.tensor(candidate_vector)
                                    ).item()
                                    if score > max_score:
                                        max_score = score
                                        best_candidate = person
                                        best_mention = mention
                                        best_sent = context

                        if best_candidate == identifier:
                            tp += 1
                            tps.append({"true": name,
                                        "pred": best_mention,
                                        "true_id": identifier,
                                        "pred_id": best_candidate,
                                        "score": max_score,
                                        "sentence": " ".join([token[0] for token in sentence]),
                                        "context": " ".join([token[0] for token in best_sent])})
                        else:
                            fp += 1
                            fps.append({"true": name,
                                        "pred": best_mention,
                                        "true_id": identifier,
                                        "pred_id": best_candidate,
                                        "score": max_score,
                                        "sentence": " ".join([token[0] for token in sentence]),
                                        "context": " ".join([token[0] for token in best_sent])})
            print({"accuracy": self.accuracy(tp, fp), "precision": self.precision(tp, fp)})
            stats.append(
                {"accuracy": self.accuracy(tp, fp), "precision": self.precision(tp, fp)}
            )
        return pd.DataFrame(stats).describe()
