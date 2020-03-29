import itertools
import json
from pathlib import Path

import flair
import torch
import tqdm
from flair.data import Sentence, DataPair
from flair.datasets import FlairDataset
from flair.embeddings import BertEmbeddings, DocumentRNNEmbeddings
from flair.models.similarity_learning_model import (
    CosineSimilarity,
    RankingLoss,
    SimilarityLearner,
)
from flair.trainers import ModelTrainer


class FaktotumDataset(FlairDataset):
    def __init__(self, name: str, in_memory: bool = True, **kwargs):
        super(FaktotumDataset, self).__init__()
        self.name = name
        self.train = list()
        self.dev = list()
        self.test = list()

        for instance in tqdm.tqdm(self._load_corpus("dev")):
            sentence = Sentence(instance["sentence"], use_tokenizer=False)
            context = Sentence(instance["context"], use_tokenizer=False)
            sentence.person = instance["person"]
            context.person = instance["person"]
            point = DataPair(sentence, context)
            self.train.append(point)

        for instance in tqdm.tqdm(self._load_corpus("dev")):
            sentence = Sentence(instance["sentence"], use_tokenizer=False)
            context = Sentence(instance["context"], use_tokenizer=False)
            sentence.person = instance["person"]
            context.person = instance["person"]
            point = DataPair(sentence, context)
            self.test.append(point)

        for instance in tqdm.tqdm(self._load_corpus("dev")):
            sentence = Sentence(instance["sentence"], use_tokenizer=False)
            context = Sentence(instance["context"], use_tokenizer=False)
            sentence.person = instance["person"]
            context.person = instance["person"]
            point = DataPair(sentence, context)
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

    def __getitem__(self, index: int = 0):
        return self.data_points[index]


class SentenceSimilarity(SimilarityLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_loss(self, data_points):
        mapped_source_embeddings = self._embed_source(data_points)
        mapped_target_embeddings = self._embed_target(data_points)

        similarity_matrix = self.similarity_measure.forward(
            (mapped_source_embeddings, mapped_target_embeddings)
        )

        sources = list()
        targets = list()
        y = list()

        for a in data_points:
            for b in data_points:
                sources.append(a.first.embedding)
                targets.append(b.second.embedding)
                if a.first.person == b.second.person:
                    y.append(1.0)
                else:
                    y.append(-1.0)

        print(sources)

        sources = torch.tensor(sources).to(flair.device)
        targets = torch.tensor(targets).to(flair.device)
        y = torch.tensor(y).to(flair.device)

        loss = self.similarity_loss(sources, targets, y)

        return loss


def train(corpus_name="droc", embeddings_path="/mnt/data/users/simmler/model-zoo/ner-droc"):
    corpus = FaktotumDataset(corpus_name)

    embedding = DocumentRNNEmbeddings(
        [BertEmbeddings(embeddings_path),],
        bidirectional=True,
        dropout=0.25,
        rnn_type="LSTM",
        hidden_size=256,
    )

    source_embedding = embedding
    target_embedding = embedding

    similarity_measure = CosineSimilarity()

    similarity_loss = torch.nn.CosineEmbeddingLoss(margin=0.15)

    similarity_model = SentenceSimilarity(
        source_embeddings=source_embedding,
        target_embeddings=target_embedding,
        similarity_measure=similarity_measure,
        similarity_loss=similarity_loss,
    )

    trainer: ModelTrainer = ModelTrainer(
        similarity_model, corpus, optimizer=torch.optim.SGD
    )

    trainer.train(
        f"{corpus_name}-similarity-model",
        learning_rate=2,
        mini_batch_size=64,
        max_epochs=100,
        min_learning_rate=1e-6,
        shuffle=True,
        anneal_factor=0.5,
        patience=4,
        embeddings_storage_mode="none",
    )


if __name__ == "__main__":
    train()
    # https://omoindrot.github.io/triplet-loss#triplet-mining
    # https://gombru.github.io/2019/04/03/ranking_loss/
