import itertools
import json
from pathlib import Path

import flair
import numpy as np
import pandas as pd
import torch
import tqdm
from flair.data import DataPair, Sentence
from flair.datasets import DataLoader, FlairDataset
from flair.embeddings import BertEmbeddings, DocumentRNNEmbeddings
from flair.models.similarity_learning_model import (
    CosineSimilarity,
    RankingLoss,
    SimilarityLearner,
)
from flair.trainers import ModelTrainer
from flair.training_utils import Result, store_embeddings


class FaktotumDataset(FlairDataset):
    def __init__(self, name: str, in_memory: bool = True, **kwargs):
        super(FaktotumDataset, self).__init__()
        self.name = name
        self.train = list()
        self.dev = list()
        self.test = list()

        for instance in tqdm.tqdm(self._load_corpus("test")):
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


class SentenceSimilarityLearner(SimilarityLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_loss(self, data_points):
        source_embeddings = self._embed_source(data_points)
        target_embeddings = self._embed_target(data_points)

        sources = list()
        targets = list()
        y = list()
        _sources = set()

        for i, a in enumerate(data_points):
            for j, b in enumerate(data_points):
                if str(a.first) not in _sources:
                    sources.append(source_embeddings[i])
                    targets.append(target_embeddings[j])
                    if a.first.person == b.second.person:
                        y.append(1.0)
                    else:
                        y.append(-1.0)
                    _sources.add(str(a.first))

        sources = torch.stack(sources).to(flair.device)
        targets = torch.stack(targets).to(flair.device)
        y = torch.tensor(y).to(flair.device)

        loss = self.similarity_loss(sources, targets, y)

        return loss

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode="none",
    ) -> (Result, float):
        ranks = list()
        with torch.no_grad():
            targets = list()
            targets_y = list()
            for data_points in data_loader:
                targets.extend(self._embed_target(data_points).to(self.eval_device))
                targets_y.extend([sentence.second.person for sentence in data_points])
                store_embeddings(data_points, embedding_storage_mode)

            for data_points in data_loader:
                sources = self._embed_source(data_points).to(self.eval_device)
                sources_y = [sentence.first.person for sentence in data_points]

                scores = list()
                agreement = list()
                for source, source_y in zip(sources, sources_y):
                    for target, target_y in zip(targets, targets_y):
                        score = self.similarity_measure(source, target).item()
                        scores.append(score)
                        agreement.append(source_y == target_y)

                df = pd.DataFrame({"scores": scores, "agreement": agreement})
                df = df.sort_values("scores", ascending=False).reset_index(drop=True)
                df = df[df["agreement"] == True]
                rank = min(df.index)
                ranks.append(1 - (rank / len(targets_y)))

        results_header_str = "\t".join(
            ["Median rank", "Mean rank", "Standard deviation"]
        )
        epoch_results_str = "\t".join(
            [str(np.median(ranks)), str(np.mean(ranks)), str(np.std(ranks))]
        )
        return (
            Result(np.mean(ranks), results_header_str, epoch_results_str, "",),
            0,
        )


def train(
    corpus_name="droc", embeddings_path="/mnt/data/users/simmler/model-zoo/ner-droc"
):
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

    similarity_measure = torch.nn.CosineSimilarity(dim=-1)

    similarity_loss = torch.nn.CosineEmbeddingLoss(margin=0.15)

    similarity_model = SentenceSimilarityLearner(
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
