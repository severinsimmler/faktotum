import itertools
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import flair
import torch
from flair.data import DataPair, DataPoint, Sentence
from flair.datasets import FlairDataset
from flair.embeddings import BertEmbeddings, DocumentRNNEmbeddings
from flair.models.similarity_learning_model import (
    RankingLoss,
    SimilarityLearner,
    SimilarityMeasure,
    CosineSimilarity
)
from flair.trainers import ModelTrainer
from torch.autograd import Variable
from abc import abstractmethod

import flair
from flair.data import DataPoint, DataPair
from flair.embeddings import Embeddings
from flair.datasets import DataLoader
from flair.training_utils import Result
from flair.training_utils import store_embeddings

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import itertools

from typing import Union, List
from pathlib import Path
import tqdm


class FaktotumDataset(FlairDataset):
    def __init__(self, name: str, in_memory: bool = True, **kwargs):
        super(FaktotumDataset, self).__init__()
        self.name = name
        self.train = list()
        self.dev = list()
        self.test = list()

        print("Train")
        for instance in tqdm.tqdm(self._load_corpus("test")):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.entity_indices = instance["sentence_indices"]
            a.identifier = instance["sentence_identifier"]
            b.entity_indices = instance["context_indices"]
            b.identifier = instance["context_identifier"]
            point = DataPair(a, b)
            point.similar = instance["similar"]
            self.train.append(point)

        print("Test")
        for instance in tqdm.tqdm(self._load_corpus("dev")):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.entity_indices = instance["sentence_indices"]
            a.identifier = instance["sentence_identifier"]
            b.entity_indices = instance["context_indices"]
            b.identifier = instance["context_identifier"]
            point = DataPair(a, b)
            point.similar = instance["similar"]
            self.test.append(point)

        print("Dev")
        for instance in tqdm.tqdm(self._load_corpus("dev")):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.entity_indices = instance["sentence_indices"]
            a.identifier = instance["sentence_identifier"]
            b.entity_indices = instance["context_indices"]
            b.identifier = instance["context_identifier"]
            point = DataPair(a, b)
            point.similar = instance["similar"]
            self.dev.append(point)

        self.data_points = self.train + self.test + self.dev
        self.train = self.train[:200]
        self.test = self.test[:50]
        self.dev = self.dev[:50]

    def _load_corpus(self, dataset):
        module = Path(__file__).resolve().parent
        data = Path(
            module, "data", self.name, "similarity", f"{dataset}.json"
        ).read_text(encoding="utf-8")
        return json.loads(data)

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index: int = 0) -> DataPair:
        return self.data_points[index]


class EntitySimilarity(SimilarityLearner):
    def __init__(self, **kwargs):
        super(EntitySimilarity, self).__init__(**kwargs)

    @staticmethod
    def _average_vectors(vectors):
        vector = vectors[0]
        for v in vectors[1:]:
            vector = vector + v
        return vector / len(vectors)

    def _embed_entities(self, data_points):
        self.source_embeddings.embed(data_points)

        entities = list()
        for sentence in data_points:
            entity = [sentence[index].embedding for index in sentence.entity_indices]
            entity = self._average_vectors(entity)
            entities.append(entity)
        entities = torch.stack(entities).to(flair.device)
        return Variable(entities, requires_grad=True)

    @staticmethod
    def _get_y(data_points):
        return torch.tensor([sentence.similar for sentence in data_points]).to(flair.device)

    def forward_loss(
        self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        source = self._embed_source([point.first for point in data_points])
        target = self._embed_target([point.second for point in data_points])
        y = self._get_y(data_points)
        return self.similarity_loss(source, target, y)

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode="none",
    ) -> (Result, float):
        tp = 0
        fp = 0
        with torch.no_grad():
            for batch in data_loader:
            data_points = [data_point for data_point in batch if data_point.similar == 1]
            
            sources = list()
            sources_ = set()
            targets = list()
            targets_ = set()
            for point in data_points:
                if point.first.identifier not in sources_:
                    sources.append(point.first)
                    sources_.add(point.first.identifier)
                if point.second.identifier not in targets_:
                    targets.append(point.second)
                    targets_.add(point.second.identifier)

            sources = self._embed_entities(sources).to(self.eval_device)
            targets = self._embed_entities(targets).to(self.eval_device)

            print("Evaluating")
            for source in tqdm.tqdm(sources):
                best_score = 0.0
                best_label = None
                for target in targets:
                    score = self.similarity_measure(source, target).item()
                    if score > best_score:
                        best_score = score
                        best_label = target.person
                if best_label == source.person:
                    tp += 1
                else:
                    fp += 1
        
        precision = tp / (tp + fp)
        print("PRECISION", precision)

        return (
            Result(
                precision,
                f"{precision}",
                f"{precision}",
                f"{precision}",
            ),
            0,
        )



def test():
    corpus = FaktotumDataset("droc")
    embedding = DocumentRNNEmbeddings(
        [
            BertEmbeddings(
        "/mnt/data/users/simmler/model-zoo/ner-droc"
        ),
        ],
        bidirectional=True,
        dropout=0.25,
        hidden_size=256,
    )

    similarity_measure = CosineSimilarity()

    similarity_loss = torch.nn.CosineEmbeddingLoss()

    similarity_model = EntitySimilarity(
        source_embeddings=embedding,
        target_embeddings=embedding,
        similarity_measure=similarity_measure,
        similarity_loss=similarity_loss,
    )

    trainer: ModelTrainer = ModelTrainer(
        similarity_model, corpus, optimizer=torch.optim.SGD
    )

    trainer.train(
        "smartdata-cosine-bcp-improved-loss",
        mini_batch_size=32,
        embeddings_storage_mode="none",
    )


if __name__ == "__main__":
    test()
