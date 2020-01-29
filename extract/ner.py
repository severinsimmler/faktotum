import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import sklearn.model_selection
import torch
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from extract.corpus import Token
from extract.evaluation import evaluate

Dataset = List[List[Token]]


@dataclass
class Baseline:
    train: Dataset
    val: Dataset
    test: Dataset

    def __post_init__(self):
        self._translate_labels(self.train)
        self._translate_labels(self.val)
        self._translate_labels(self.test)

    def _translate_labels(self, data):
        for sentence in data:
            previous = "[START]"
            for token in sentence:
                if (
                    previous == "I-FIRST_NAME" or previous == "B-FIRST_NAME"
                ) and token.label == "B-LAST_NAME":
                    label = "I-PER"
                else:
                    label = self.custom2germeval.get(token.label, "O")
                previous = token.label
                token.label = label

    def _load_custom_dataset(self):
        _train = Path(tempfile.NamedTemporaryFile().name)
        _test = Path(tempfile.NamedTemporaryFile().name)
        _val = Path(tempfile.NamedTemporaryFile().name)

        with _train.open("w", encoding="utf-8") as file_:
            sentences = [
                "\n".join([f"{token.text} {token.label}" for token in sentence]).strip()
                for sentence in self.train
            ]
            file_.write("\n\n".join(sentences))
        with _test.open("w", encoding="utf-8") as file_:
            sentences = [
                "\n".join([f"{token.text} {token.label}" for token in sentence]).strip()
                for sentence in self.test
            ]
            file_.write("\n\n".join(sentences))
        with _val.open("w", encoding="utf-8") as file_:
            sentences = [
                "\n".join([f"{token.text} {token.label}" for token in sentence]).strip()
                for sentence in self.val
            ]
            file_.write("\n\n".join(sentences))

        corpus = ColumnCorpus(
            _train.parent,
            {0: "text", 1: "ner"},
            train_file=_train.name,
            test_file=_test.name,
            dev_file=_val.name,
        )
        _train.unlink()
        _test.unlink()
        _val.unlink()
        return corpus

    def _load_germeval_dataset(self):
        current_folder = Path(__file__).parent
        data_folder = Path(current_folder, "data", "germeval")
        columns = {1: "text", 2: "ner"}
        return ColumnCorpus(
            data_folder,
            columns,
            train_file="train.tsv",
            test_file="test.tsv",
            dev_file="dev.tsv",
        )

    def _train_flair_model(self, name, corpus, tagger=None):
        tag_dictionary = corpus.make_tag_dictionary(tag_type="ner")
        if not tagger:
            tagger = SequenceTagger(
                hidden_size=256,
                embeddings=[PooledFlairEmbeddings("news-forward")],
                tag_dictionary=tag_dictionary,
                tag_type=tag_type,
                use_crf=True,
            )
        trainer = ModelTrainer(tagger, corpus)
        trainer.train(name, learning_rate=0.1, mini_batch_size=32, max_epochs=50)
        return Path(name, "final-model.pt")

    def scratch(self):
        """Train from scratch _only_ on custom dataset."""
        corpus = self._load_custom_dataset()
        model_path = _train_flair_model("scratch", corpus)
        tagger = SequenceTagger.load(model_path)
        pred = self._prediction(tagger)
        metric = evaluate("scratch", self.test, pred)
        print(metric)
        return metric

    def vanilla(self):
        """Train from scratch _only_ on Germeval dataset and evaluate on custom dataset."""
        corpus = self._load_germeval_dataset()
        model_path = _train_flair_model("vanilla", corpus)
        tagger = SequenceTagger.load(model_path)
        pred = self._prediction(tagger)
        metric = evaluate(name, self.test, pred)
        print(metric)
        return metric

    def continued(self, model_path):
        """Continue training with custom dataset."""
        corpus = self._load_custom_dataset()
        tagger = SequenceTagger.load(model_path)
        model_path = _train_flair_model("continued", corpus, tagger)
        tagger = SequenceTagger.load(model_path)
        pred = self._prediction(tagger)
        metric = evaluate(f"{name}-continued", self.test, pred)
        print(metric)
        return metric

    def _prediction(self, tagger: SequenceTagger) -> Dataset:
        preds = list()
        for sentence in self.test:
            text = " ".join([token.text for token in sentence])
            sentence = Sentence(text, use_tokenizer=False)
            tagger.predict(sentence)
            pred = [
                Token(
                    token.text,
                    index,
                    self.germeval2custom.get(token.get_tag("ner").value, "O"),
                )
                for index, token in enumerate(sentence)
            ]
            preds.append(pred)
        return preds

    @property
    def custom2germeval(self):
        return {
            "B-FIRST_NAME": "B-PER",
            "I-FIRST_NAME": "I-PER",
            "B-LAST_NAME": "B-PER",
            "I-LAST_NAME": "I-PER",
            "B-ORGANIZATION": "B-ORG",
            "I-ORGANIZATION": "I-ORG",
        }

    @property
    def germeval2custom(self):
        return {
            "B-PER": "B-PER",
            "I-PER": "I-PER",
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
        }


@dataclass
class ConditionalRandomField:
    train: Dataset
    val: Dataset
    test: Dataset

    def __post_init__(self):
        self._translate_labels(self.train)
        self._translate_labels(self.val)
        self._translate_labels(self.test)


@dataclass
class RuleBased:
    train: Dataset
    val: Dataset
    test: Dataset

    def __post_init__(self):
        module_folder = Path(__file__).resolve().parent
        with Path(module_folder, "data", "persons.json").open(
            "r", encoding="utf-8"
        ) as file_:
            self.persons = json.load(file_)
        with Path(module_folder, "data", "organizations.json").open(
            "r", encoding="utf-8"
        ) as file_:
            self.organizations = json.load(file_)

    def __post_init__(self):
        self._translate_labels(self.train)
        self._translate_labels(self.val)
        self._translate_labels(self.test)

    def vanilla(self):
        preds = list()
        for sentence in self.test:
            pred = list()
            previous = "[START]"
            for token in sentence:
                if token.text in self.persons and (
                    previous == "B-PER" or previous == "I-PER"
                ):
                    label = "I-PER"
                elif token.text in self.persons and (
                    previous != "B-PER" or previous != "I-PER"
                ):
                    label = "B-PER"
                elif token.text in self.organizations and (
                    previous == "B-ORG" or previous == "I-ORG"
                ):
                    label = "I-PER"
                elif token.text in self.organizations and (
                    previous != "B-ORG" or previous != "I-ORG"
                ):
                    label = "B-PER"
                else:
                    label = "O"
                pred.append(Token(token.text, token.index, label))
            preds.append(pred)
        metric = evaluate(f"vanilla-rule-based", self.test, preds)
        print(metric)
        return metric
