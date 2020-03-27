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
    PairwiseBCELoss,
    SimilarityLearner,
    SimilarityMeasure,
    ModelSimilarity
)
from flair.trainers import ModelTrainer
from torch.autograd import Variable


class FaktotumDataset(FlairDataset):
    def __init__(self, name: str, in_memory: bool = True, **kwargs):
        super(FaktotumDataset, self).__init__()
        self.name = name
        self.train = list()
        self.dev = list()
        self.test = list()

        for instance in self._load_corpus("train"):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.INDEX = instance["sentence_index"]
            b.INDEX = instance["context_index"]
            a.ID = instance["sentence_id"]
            b.ID = instance["context_id"]
            point = DataPair(a, b)
            point.ID = instance["person_id"]
            self.train.append(point)

        for instance in self._load_corpus("test"):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.INDEX = instance["sentence_index"]
            b.INDEX = instance["context_index"]
            a.ID = instance["sentence_id"]
            b.ID = instance["context_id"]
            point = DataPair(a, b)
            point.ID = instance["person_id"]
            self.test.append(point)

        for instance in self._load_corpus("dev"):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.INDEX = instance["sentence_index"]
            b.INDEX = instance["context_index"]
            a.ID = instance["sentence_id"]
            b.ID = instance["context_id"]
            point = DataPair(a, b)
            point.ID = instance["person_id"]
            self.dev.append(point)

        self.data_points = self.train + self.test + self.dev

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


class CosineSimilarity(SimilarityMeasure):
    def forward(self, x):
        input_modality_0 = x[0]
        input_modality_1 = x[1]
        cos = torch.nn.CosineSimilarity(dim=-1)
        return cos(input_modality_0, input_modality_1)



class EntitySimilarityLearner(SimilarityLearner):
    def __init__(self, **kwargs):
        super(EntitySimilarityLearner, self).__init__(**kwargs)

    def _embed_source(self, data_points):
        data_points = [point.first for point in data_points]

        self.source_embeddings.embed(data_points)

        source_embedding_tensor = torch.stack(
            [point[point.INDEX].embedding for point in data_points]
        ).to(flair.device)

        if self.source_mapping is not None:
            source_embedding_tensor = self.source_mapping(source_embedding_tensor)

        return Variable(source_embedding_tensor, requires_grad=True)

    def _embed_target(self, data_points):

        data_points = [point.second for point in data_points]

        self.target_embeddings.embed(data_points)

        target_embedding_tensor = torch.stack(
            [point[point.INDEX].embedding for point in data_points]
        ).to(flair.device)

        if self.target_mapping is not None:
            target_embedding_tensor = self.target_mapping(target_embedding_tensor)

        return Variable(target_embedding_tensor, requires_grad=True)

    def forward_loss(
        self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        mapped_source_embeddings = self._embed_source(data_points)
        mapped_target_embeddings = self._embed_target(data_points)

        if self.interleave_embedding_updates:
            # 1/3 only source branch of model, 1/3 only target branch of model, 1/3 both
            detach_modality_id = torch.randint(0, 3, (1,)).item()
            if detach_modality_id == 0:
                mapped_source_embeddings.detach()
            elif detach_modality_id == 1:
                mapped_target_embeddings.detach()

        similarity_matrix = self.similarity_measure.forward(
            (mapped_source_embeddings, mapped_target_embeddings)
        )

        def add_to_index_map(hashmap, key, val):
            if key not in hashmap:
                hashmap[key] = [val]
            else:
                hashmap[key] += [val]

        index_map = {"first": {}, "second": {}}
        person_indices = defaultdict(list)
        for data_point_id, data_point in enumerate(data_points):
            add_to_index_map(index_map["first"], data_point.first.ID, data_point_id)
            add_to_index_map(index_map["second"], data_point.second.ID, data_point_id)
            person_indices[data_point.ID].append(data_point_id)

        for key, value in person_indices.items():
            for i in value:
                for sent_id, point_ids in index_map["first"].items():
                    if i in point_ids:
                        index_map["first"][sent_id].extend(value)
                        index_map["first"][sent_id] = sorted(
                            list(set(index_map["first"][sent_id]))
                        )
                for sent_id, point_ids in index_map["second"].items():
                    if i in point_ids:
                        index_map["second"][sent_id].extend(value)
                        index_map["second"][sent_id] = sorted(
                            list(set(index_map["second"][sent_id]))
                        )

        targets = torch.zeros_like(similarity_matrix).to(flair.device)

        for data_point in data_points:
            first_indices = index_map["first"][data_point.first.ID]
            second_indices = index_map["second"][data_point.second.ID]
            for first_index, second_index in itertools.product(
                first_indices, second_indices
            ):
                targets[first_index, second_index] = 1.0

        loss = self.similarity_loss(similarity_matrix, targets)

        return loss


def test():
    corpus = FaktotumDataset()
    embedding = BertEmbeddings(
        "/mnt/data/users/simmler/model-zoo/bert-multi-presse-adapted"
    )

    source_embedding = embedding
    target_embedding = embedding

    similarity_measure = ModelSimilarity()

    similarity_loss = PairwiseBCELoss()

    similarity_model = EntitySimilarityLearner(
        source_embeddings=source_embedding,
        target_embeddings=target_embedding,
        similarity_measure=similarity_measure,
        similarity_loss=similarity_loss,
    )

    trainer: ModelTrainer = ModelTrainer(
        similarity_model, corpus, optimizer=torch.optim.SGD
    )

    trainer.train(
        "smartdata-cosine-bcp-improved-loss",
        mini_batch_size=16,
        embeddings_storage_mode="none",
    )


if __name__ == "__main__":
    test()
