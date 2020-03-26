import json
from pathlib import Path

import torch
from torch.autograd import Variable
import flair
from flair.data import DataPair, DataPoint, Sentence
from flair.datasets import FlairDataset
from flair.embeddings import DocumentRNNEmbeddings, BertEmbeddings
from flair.models.similarity_learning_model import SimilarityLearner, CosineSimilarity, RankingLoss
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
            point = DataPair(a, b)
            self.train.append(point)

        for instance in self._load_corpus("dev"):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.INDEX = instance["sentence_index"]
            b.INDEX = instance["context_index"]
            point = DataPair(a, b)
            self.test.append(point)

        for instance in self._load_corpus("dev"):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.INDEX = instance["sentence_index"]
            b.INDEX = instance["context_index"]
            point = DataPair(a, b)
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


def test():
    corpus = FaktotumDataset()
    embedding = BertEmbeddings('/mnt/data/users/simmler/model-zoo/ner-droc')

    source_embedding = embedding
    target_embedding = embedding

    similarity_measure = CosineSimilarity()

    similarity_loss = RankingLoss(margin=0.15)

    similarity_model = EntitySimilarityLearner(source_embeddings=source_embedding,
                                               target_embeddings=target_embedding,
                                               similarity_measure=similarity_measure,
                                               similarity_loss=similarity_loss)

    print(similarity_model)

    trainer: ModelTrainer = ModelTrainer(similarity_model, corpus, optimizer=torch.optim.SGD)

    trainer.train(
        'droc-cosine-ranking',
        learning_rate=2,
        mini_batch_size=128,
        max_epochs=1000,
        min_learning_rate=1e-6,
        shuffle=True,
        anneal_factor=0.5,
        patience=4,
        embeddings_storage_mode='none'
    )

"""
import json
from pathlib import Path

import torch

import flair
from flair.data import DataPair, DataPoint, Sentence
from flair.datasets import FlairDataset
from flair.embeddings import DocumentRNNEmbeddings, BertEmbeddings
from flair.models.similarity_learning_model import SimilarityLearner, CosineSimilarity, RankingLoss
from flair.trainers import ModelTrainer
from faktotum.similarity import EntitySimilarityLearner, FaktotumDataset

corpus = FaktotumDataset()
embedding = BertEmbeddings('/mnt/data/users/simmler/model-zoo/ner-droc')

source_embedding = embedding
target_embedding = embedding

similarity_measure = CosineSimilarity()

similarity_loss = RankingLoss(margin=0.15)

similarity_model = EntitySimilarityLearner(source_embeddings=source_embedding,
                                            target_embeddings=target_embedding,
                                            similarity_measure=similarity_measure,
                                            similarity_loss=similarity_loss)

print(similarity_model)

trainer: ModelTrainer = ModelTrainer(similarity_model,corpus, optimizer=torch.optim.SGD)

trainer.train(
    'droc-cosine-ranking',
    learning_rate=2,
    mini_batch_size=32,
    max_epochs=1000,
    min_learning_rate=1e-6,
    shuffle=True,
    anneal_factor=0.5,
    patience=4,
    embeddings_storage_mode='none'
)
"""