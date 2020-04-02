import itertools
import json
from pathlib import Path
from typing import List, Union

import flair
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from flair.data import DataPair, DataPoint, Dictionary, Image, Sentence, Token
from flair.datasets import DataLoader, FlairDataset
from flair.embeddings import BertEmbeddings, DocumentRNNEmbeddings, Embeddings
from flair.file_utils import cached_path, open_inside_zip
from flair.models.similarity_learning_model import (
    CosineSimilarity,
    RankingLoss,
    SimilarityLearner,
)
from flair.nn import LockedDropout, WordDropout
from flair.trainers import ModelTrainer
from flair.training_utils import Result, store_embeddings
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FaktotumDataset(FlairDataset):
    def __init__(self, name: str, in_memory: bool = True, **kwargs):
        super(FaktotumDataset, self).__init__()
        self.name = name
        self.train = list()
        self.dev = list()
        self.test = list()

        for instance in tqdm.tqdm(self._load_corpus("train")):
            sentence = Sentence(instance["sentence"], use_tokenizer=False)
            context = Sentence(instance["context"], use_tokenizer=False)
            sentence.person = instance["person"]
            sentence.indices = instance["sentence_indices"]
            context.person = instance["person"]
            context.indices = instance["context_indices"]
            point = DataPair(sentence, context)
            self.train.append(point)

        for instance in tqdm.tqdm(self._load_corpus("test")):
            sentence = Sentence(instance["sentence"], use_tokenizer=False)
            context = Sentence(instance["context"], use_tokenizer=False)
            sentence.person = instance["person"]
            sentence.indices = instance["sentence_indices"]
            context.person = instance["person"]
            context.indices = instance["context_indices"]
            point = DataPair(sentence, context)
            self.test.append(point)

        for instance in tqdm.tqdm(self._load_corpus("dev")):
            sentence = Sentence(instance["sentence"], use_tokenizer=False)
            context = Sentence(instance["context"], use_tokenizer=False)
            sentence.person = instance["person"]
            sentence.indices = instance["sentence_indices"]
            context.person = instance["person"]
            context.indices = instance["context_indices"]
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

    def __repr__(self):
        return f"<FaktotumCorpus: {len(self.train)} train, {len(self.test)} test, {len(self.dev)} dev>"

    def __str__(self):
        return f"<FaktotumCorpus: {len(self.train)} train, {len(self.test)} test, {len(self.dev)} dev>"


class EntityEmbeddings(DocumentRNNEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def embed(self, sentences, indices) -> List[Sentence]:
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

    def _add_embeddings_internal(self, sentences, indices):
        if not hasattr(self, "locked_dropout"):
            self.locked_dropout = None
        if not hasattr(self, "word_dropout"):
            self.word_dropout = None

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

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
            all_embs += [
                emb
                for i, token in enumerate(sentence)
                for emb in token.get_each_embedding()
                if i in index
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(index)

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


class EntitySimilarityLearner(SimilarityLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _embed_source(self, data_points):
        indices = [point.first.indices for point in data_points]
        data_points = [point.first for point in data_points]

        self.source_embeddings.embed(data_points, indices)

        source_embedding_tensor = torch.stack(
            [point.embedding for point in data_points]
        ).to(flair.device)

        return source_embedding_tensor

    def _embed_target(self, data_points):
        indices = [point.second.indices for point in data_points]
        data_points = [point.second for point in data_points]

        self.target_embeddings.embed(data_points, indices)

        target_embedding_tensor = torch.stack(
            [point.embedding for point in data_points]
        ).to(flair.device)

        return target_embedding_tensor

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

        return self.similarity_loss(sources, targets, y)

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode="none",
    ) -> (Result, float):
        ranks_min = list()
        ranks_max = list()

        with torch.no_grad():
            targets = list()
            targets_y = list()
            target_sentences = list()
            for data_points in data_loader:
                targets.extend(
                    [
                        tensor
                        for tensor in self._embed_target(data_points).to(
                            self.eval_device
                        )
                    ]
                )
                targets_y.extend([sentence.second.person for sentence in data_points])
                target_sentences.extend(
                    [str(sentence.second) for sentence in data_points]
                )
                store_embeddings(data_points, embedding_storage_mode)

            for data_points in data_loader:
                sources = self._embed_source(data_points).to(self.eval_device)
                sources_y = [sentence.first.person for sentence in data_points]
                source_sentences = [str(sentence.first) for sentence in data_points]

                scores = list()
                agreement = list()
                for source, source_y, source_sentence in zip(
                    sources, sources_y, source_sentences
                ):
                    for target, target_y, target_sentence in zip(
                        targets, targets_y, target_sentences
                    ):
                        if source_sentence != target_sentence:
                            score = self.similarity_measure(source, target).item()
                            scores.append(score)
                            agreement.append(source_y == target_y)

                df = pd.DataFrame({"scores": scores, "agreement": agreement})
                df = df.sort_values("scores", ascending=False).reset_index(drop=True)
                df = df[df["agreement"] == True]
                ranks_min.append(1 - min(df.index) / len(agreement))
                ranks_max.append(1 - max(df.index) / len(agreement))

        results_header_str = "\t".join(
            ["Median max rank", "Median min rank", "Best", "Worst"]
        )
        epoch_results_str = "\t".join(
            [
                str(np.median(ranks_max)),
                str(np.median(ranks_min)),
                str(max(ranks_min)),
                str(min(ranks_max)),
            ]
        )
        return (
            Result(np.median(ranks_max), results_header_str, epoch_results_str, "",),
            0,
        )


def train(
    corpus_name="smartdata",
    embeddings_path="/mnt/data/users/simmler/model-zoo/bert-multi-presse-adapted",
):
    corpus = FaktotumDataset(corpus_name)

    embedding = EntityEmbeddings(
        embeddings=[BertEmbeddings(embeddings_path),],
        bidirectional=True,
        dropout=0.25,
        rnn_type="GRU",
        hidden_size=256,
    )

    source_embedding = embedding
    target_embedding = embedding

    similarity_measure = torch.nn.CosineSimilarity(dim=-1)

    similarity_loss = torch.nn.CosineEmbeddingLoss(margin=0.15)

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
        f"/mnt/data/users/simmler/model-zoo/similarity-gru-{corpus_name}",
        learning_rate=2,
        mini_batch_size=32,
        max_epochs=1000,
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
