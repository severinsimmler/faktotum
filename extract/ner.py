import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import flair
import torch
from flair.data import Sentence, MultiCorpus
from flair.datasets import ColumnCorpus
from flair.embeddings import PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from extract.evaluation import evaluate_labels


@dataclass
class Baseline:
    directory: Union[str, Path]
    train_file: str
    test_file: str
    dev_file: str

    def _load_corpus(self, data_folder: Union[str, Path] = None) -> ColumnCorpus:
        return ColumnCorpus(
            data_folder=data_folder if data_folder else self.directory,
            column_format={0: "text", 1: "ner"},
            train_file=self.train_file,
            test_file=self.test_file,
            dev_file=self.dev_file,
        )

    def _train(
        self,
        output_dir: Union[str, Path],
        corpus: Optional[ColumnCorpus] = None,
        tagger: Optional[SequenceTagger] = None,
        hidden_size: int = 256,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        max_epochs: int = 100,
        use_crf: bool = True,
    ) -> SequenceTagger:
        tag_dictionary = corpus.make_tag_dictionary(tag_type="ner")
        if not tagger:
            tagger = SequenceTagger(
                hidden_size=hidden_size,
                embeddings=PooledFlairEmbeddings("news-forward"),
                tag_dictionary=tag_dictionary,
                tag_type="ner",
                use_crf=use_crf,
            )
        trainer = ModelTrainer(tagger, corpus)
        trainer.train(
            output_dir,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=max_epochs,
        )
        model_path = Path(output_dir, "best-model.pt")
        return SequenceTagger.load(model_path)

    @staticmethod
    def _parse_data(filepath):
        sentence = list()
        for row in Path(filepath).read_text(encoding="utf-8").split("\n"):
            if row.startswith("# "):
                continue
            if row != "":
                sentence.append(row.split(" ")[:2])
            else:
                yield sentence
                sentence = list()

    def _evaluate(self, name: str, tagger: SequenceTagger):
        preds = list()
        golds = list()
        test = Path(self.directory, self.test_file)
        for sentence in self._parse_data(test):
            s = Sentence(" ".join([token for token, _ in sentence]), use_tokenizer=False)
            tagger.predict(s)
            preds.append([t.get_tag("ner").value for t in s])
            golds.append([label for _, label in sentence])

        with Path("prediction.json").open("w", encoding="utf-8") as file_:
            json.dump({"gold": golds, "pred": preds}, file_, indent=2)

        return evaluate_labels(name, golds, preds)

    def from_scratch(self):
        corpus = self._load_corpus()
        tagger = self._train("from-scratch-model", corpus)
        metric = self._evaluate("from-scratch-model", tagger)
        print(metric)
        return metric

    def vanilla(self, training_corpus: str = "germeval"):
        data_dir = Path(Path(self.directory).parent, training_corpus)
        training = self._load_corpus(data_dir)
        tagger = self._train("vanilla-model", training)
        metric = self._evaluate("vanilla-model", tagger)
        print(metric)
        return metric

    def multi_corpus(self,first_corpus: str = "germeval"):
        data_dir = Path(Path(self.directory).parent, first_corpus)
        first = self._load_corpus(data_dir)
        second = self._load_corpus()
        corpus = MultiCorpus([first, second])
        tagger = self._train("multi-corpus-model", corpus)
        metric = self._evaluate("multi-corpus-model", tagger)
        print(metric)
        return metric


@dataclass
class Baseline:
    directory: Union[str, Path]
    train_file: str
    test_file: str
    dev_file: str

    @staticmethod
    def _parse_data(filepath):
        sentence = list()
        for row in Path(filepath).read_text(encoding="utf-8").split("\n"):
            if row.startswith("# "):
                continue
            if row != "":
                sentence.append(row.split(" "))
            else:
                yield sentence
                sentence = list()

    def from_scratch(self):
        train_sents = self._parse_data(Path(self.directory, self.train_file))
        test_sents = self._parse_data(Path(self.directory, self.test_file))

        X_train = [self._sent2features(s) for s in train_sents]
        y_train = [self._sent2labels(s) for s in train_sents]

        X_test = [self._sent2features(s) for s in test_sents]
        y_test = [self._sent2labels(s) for s in test_sents]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        y_pred = crf.predict(X_test)

        metric = evaluate_labels("crf-baseline", y_test, y_pred)
        print(metric)
        return metric

    def _word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][3]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features


    def _sent2features(sent):
        return [self._word2features(sent, i) for i in range(len(sent))]

    def _sent2labels(sent):
        return [label for _, label, _, _ in sent]

    def _sent2tokens(sent):
        return [token for token, _, _, _ in sent]

'''
first_corpus
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
'''
