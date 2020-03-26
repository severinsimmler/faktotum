import json
from pathlib import Path

import torch

import flair
from flair.datasets import FlairDataset
from flair.embeddings import DocumentRNNEmbeddings, BertEmbeddings
from flair.models.similarity_learning_model import SimilarityLearner, CosineSimilarity, RankingLoss
from flair.trainers import ModelTrainer


class FaktotumDataset(FlairDataset):
    def __init__(self, in_memory: bool = True, **kwargs):
        super(FaktotumDataset, self).__init__()

        self.data_points: List[DataPair] = []

        for instance in self._load_corpus():
            point = DataPair(Sentence(instance["sentence"], use_tokenizer=False), Sentence(instance["context"], use_tokenizer=False))
            point.sentence_index = instance["sentence_index"]
            point.context_index = instance["context_index"]
            self.data_points.append(point)

    @staticmethod
    def _load_corpus():
        module = Path(__file__).resolve().parent
        train = Path(module, "data", "droc", "similarity", "train.json").read_text(encoding="utf-8")
        return json.loads(train)

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
                [point[point.sentence_index].embedding for point in data_points]
            ).to(flair.device)

            if self.source_mapping is not None:
                source_embedding_tensor = self.source_mapping(source_embedding_tensor)

            return source_embedding_tensor

    def _embed_target(self, data_points):

        data_points = [point.second for point in data_points]

        self.target_embeddings.embed(data_points)

        target_embedding_tensor = torch.stack(
            [point[point.sentence_index].embedding for point in data_points]
        ).to(flair.device)

        if self.target_mapping is not None:
            target_embedding_tensor = self.target_mapping(target_embedding_tensor)

        return target_embedding_tensor


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