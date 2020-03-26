import json
from pathlib import Path
from typing import Union, List
from collections import defaultdict
import uuid
import itertools
import torch
from torch.autograd import Variable
import flair
from flair.data import DataPair, DataPoint, Sentence
from flair.datasets import FlairDataset
from flair.embeddings import DocumentRNNEmbeddings, BertEmbeddings
from flair.models.similarity_learning_model import SimilarityLearner, CosineSimilarity, PairwiseBCELoss
from flair.trainers import ModelTrainer


class FaktotumDataset(FlairDataset):
    def __init__(self, in_memory: bool = True, **kwargs):
        super(FaktotumDataset, self).__init__()

        self.train: List[DataPair] = []
        self.dev = list()
        self.test = list()

        for instance in self._load_corpus("dev"):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.INDEX = instance["sentence_index"]
            b.INDEX = instance["context_index"]
            a.ID = instance["sentence_id"]
            b.ID = instance["context_id"]
            point = DataPair(a, b)
            point.ID = instance["person_id"]
            self.train.append(point)

        for instance in self._load_corpus("dev"):
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

    @staticmethod
    def _load_corpus(name):
        module = Path(__file__).resolve().parent
        data = Path(module, "data", "droc", "similarity", f"{name}.json").read_text(encoding="utf-8")
        return json.loads(data)

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index: int = 0) -> DataPair:
        return self.data_points[index]


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

    def forward_loss(self, data_points: Union[List[DataPoint], DataPoint]) -> torch.tensor:
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
            for i in  value:
                for sent_id, point_ids in index_map["first"].items():
                    if i in point_ids:
                        index_map["first"][sent_id].extend(value)
                        index_map["first"][sent_id] = sorted(list(set(index_map["first"][sent_id])))
                        break
                for sent_id, point_ids in index_map["second"].items():
                    if i in point_ids:
                        index_map["second"][sent_id].extend(value)
                        index_map["second"][sent_id] = sorted(list(set(index_map["second"][sent_id])))
                        break

        targets = torch.zeros_like(similarity_matrix).to(flair.device)

        with open("nice.json", "w") as f:
            json.dump(index_map, f, ensure_ascii=False, indent=4)
        print("jetzt")
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
    embedding = BertEmbeddings('/mnt/data/users/simmler/model-zoo/ner-droc')

    source_embedding = embedding
    target_embedding = embedding

    similarity_measure = CosineSimilarity()

    similarity_loss = PairwiseBCELoss()

    similarity_model = EntitySimilarityLearner(source_embeddings=source_embedding,
                                                target_embeddings=target_embedding,
                                                similarity_measure=similarity_measure,
                                                similarity_loss=similarity_loss)

    trainer: ModelTrainer = ModelTrainer(similarity_model,corpus, optimizer=torch.optim.SGD)

    trainer.train(
        'TEST',
        mini_batch_size=32,
        embeddings_storage_mode='none'
    )


if __name__ == "__main__":
    test()