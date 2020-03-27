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
            a.person = instance["person"]
            b.entity_indices = instance["context_indices"]
            b.identifier = instance["context_identifier"]
            b.person = instance["person"]
            point = DataPair(a, b)
            point.similar = instance["similar"]
            self.train.append(point)

        print("Test")
        for instance in tqdm.tqdm(self._load_corpus("dev")):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.entity_indices = instance["sentence_indices"]
            a.identifier = instance["sentence_identifier"]
            a.person = instance["person"]
            b.entity_indices = instance["context_indices"]
            b.identifier = instance["context_identifier"]
            b.person = instance["person"]
            point = DataPair(a, b)
            point.similar = instance["similar"]
            self.test.append(point)

        print("Dev")
        for instance in tqdm.tqdm(self._load_corpus("dev")):
            a = Sentence(instance["sentence"], use_tokenizer=False)
            b = Sentence(instance["context"], use_tokenizer=False)
            a.entity_indices = instance["sentence_indices"]
            a.identifier = instance["sentence_identifier"]
            a.person = instance["person"]
            b.entity_indices = instance["context_indices"]
            b.identifier = instance["context_identifier"]
            b.person = instance["person"]
            point = DataPair(a, b)
            point.similar = instance["similar"]
            self.dev.append(point)

        self.data_points = self.train + self.test + self.dev
        self.train = self.train[:5000]
        self.test = self.test[:1000]
        self.dev = self.dev[:1000]

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

    def _embed_source(self, data_points):
        self.source_embeddings.embed(data_points, [index for index in sentence.entity_indices])

        entities = list()
        for sentence in data_points:
            entities.append(sentence.embedding)
        entities = torch.stack(entities).to(flair.device)
        return Variable(entities, requires_grad=True)

    def _embed_target(self, data_points):
        self.target_embeddings.embed(data_points, [index for index in sentence.entity_indices])

        entities = list()
        for sentence in data_points:
            entities.append(sentence.embedding)
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
        with torch.no_grad():
            i = 0
            score = 0.0
            for data_points in data_loader:
                source = self._embed_source([point.first for point in data_points])
                target = self._embed_target([point.second for point in data_points])
                y = self._get_y(data_points)
                score += self.similarity_loss(source, target, y).item()
                i += 1
            score = score / i
        return (
            Result(
                1 - score,
                f"{score}",
                f"{score}",
                f"{score}",
            ),
            0,
        )

class EntityEmbeddings(DocumentRNNEmbeddings):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def _add_embeddings_internal(self, sentences, indices):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        # TODO: remove in future versions
        if not hasattr(self, "locked_dropout"):
            self.locked_dropout = None
        if not hasattr(self, "word_dropout"):
            self.word_dropout = None

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

        # embed words in the sentence
        self.embeddings.embed(sentences)

        lengths: List[int] = [len(index) for index in indices]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs: List[torch.Tensor] = list()
        for index, sentence in zip(indices, sentences):
            for i, token in enumerate(sentence):
                if i in index:
                    for emb in token.get_each_embedding():
                        all_embs.append(emb)

            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        # before-RNN dropout
        if self.dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)
        if self.word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        # reproject if set
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        # push through RNN
        packed = pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )
        rnn_out, hidden = self.rnn(packed)
        outputs, output_lengths = pad_packed_sequence(rnn_out, batch_first=True)

        # after-RNN dropout
        if self.dropout:
            outputs = self.dropout(outputs)
        if self.locked_dropout:
            outputs = self.locked_dropout(outputs)

        # extract embeddings from RNN
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[sentence_no, length - 1]

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[sentence_no, 0]
                embedding = torch.cat([first_rep, last_rep], 0)

            if self.static_embeddings:
                embedding = embedding.detach()

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def embed(self, sentences: Union[Sentence, List[Sentence]], indices) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if (type(sentences) is Sentence) or (type(sentences) is Image):
            sentences = [sentences]

        everything_embedded: bool = True

        if self.embedding_type == "word-level":
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences, indices)

        return sentences


def test():
    corpus = FaktotumDataset("droc")
    embedding = EntityEmbeddings(
        [
            BertEmbeddings(
        "/mnt/data/users/simmler/model-zoo/ner-droc"
        ),
        ],
        bidirectional=True,
        dropout=0.25,
        hidden_size=256,
        rnn_type="LSTM"
    )

    similarity_measure = torch.nn.CosineSimilarity(dim=-1)

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
