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
from flair.embeddings import BertEmbeddings, DocumentRNNEmbeddings, DocumentEmbeddings
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


import os
import re
import logging
from abc import abstractmethod
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Dict, Tuple

import hashlib

import gensim
import numpy as np
import torch
from bpemb import BPEmb
from deprecated import deprecated

import torch.nn.functional as F
from torch.nn import ParameterList, Parameter
from torch.nn import Sequential, Linear, Conv2d, ReLU, MaxPool2d, Dropout2d
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from transformers import (
    AlbertTokenizer,
    AlbertModel,
    BertTokenizer,
    BertModel,
    CamembertTokenizer,
    CamembertModel,
    RobertaTokenizer,
    RobertaModel,
    TransfoXLTokenizer,
    TransfoXLModel,
    OpenAIGPTModel,
    OpenAIGPTTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    XLNetTokenizer,
    XLMTokenizer,
    XLNetModel,
    XLMModel,
    XLMRobertaTokenizer,
    XLMRobertaModel,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import flair
from flair.data import Corpus
from flair.nn import LockedDropout, WordDropout
from flair.data import Dictionary, Token, Sentence, Image
from flair.file_utils import cached_path, open_inside_zip
from flair.embeddings import *

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


class _DocumentRNNEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        embeddings: List[TokenEmbeddings],
        hidden_size=128,
        rnn_layers=1,
        reproject_words: bool = True,
        reproject_words_dimension: int = None,
        bidirectional: bool = False,
        dropout: float = 0.5,
        word_dropout: float = 0.0,
        locked_dropout: float = 0.0,
        rnn_type="GRU",
        fine_tune: bool = True,
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the rnn
        :param rnn_layers: the number of layers for the rnn
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        :param rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.rnn_type = rnn_type

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False if fine_tune else True

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )

        # bidirectional RNN on top of embedding layer
        if rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
        elif rnn_type == "Transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.embeddings_dimension, nhead=8)
            self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=6)
        else:
            self.rnn = torch.nn.GRU(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            )

        self.name = "document_" + self.rnn._get_name()

        # dropouts
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.locked_dropout = (
            LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        )
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: Union[List[Sentence], Sentence]):
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

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs: List[torch.Tensor] = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
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

    def _apply(self, fn):
        major, minor, build, *_ = (int(info)
                                for info in torch.__version__.replace("+",".").split('.') if info.isdigit())

        # fixed RNN change format for torch 1.4.0
        if major >= 1 and minor >= 4:
            for child_module in self.children():
                if isinstance(child_module, torch.nn.RNNBase):
                    _flat_weights_names = []
                    num_direction = None

                    if child_module.__dict__["bidirectional"]:
                        num_direction = 2
                    else:
                        num_direction = 1
                    for layer in range(child_module.__dict__["num_layers"]):
                        for direction in range(num_direction):
                            suffix = "_reverse" if direction == 1 else ""
                            param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                            if child_module.__dict__["bias"]:
                                param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                            param_names = [
                                x.format(layer, suffix) for x in param_names
                            ]
                            _flat_weights_names.extend(param_names)

                    setattr(child_module, "_flat_weights_names",
                            _flat_weights_names)

                child_module._apply(fn)

        else:
            super()._apply(fn)



def test():
    corpus = FaktotumDataset("droc")
    embedding = _DocumentRNNEmbeddings(
        [
            BertEmbeddings(
        "/mnt/data/users/simmler/model-zoo/ner-droc"
        ),
        ],
        bidirectional=True,
        dropout=0.25,
        hidden_size=256,
        rnn_type="Transformer"
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
