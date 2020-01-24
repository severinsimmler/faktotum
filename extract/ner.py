import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import sklearn.model_selection
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from extract.corpus import Token
from extract.evaluation import evaluate

Dataset = List[List[Token]]


@dataclass
class Baseline:
    train: Dataset
    test: Dataset

    def _load_dataset(self):
        _train = Path(tempfile.NamedTemporaryFile().name)
        _test = Path(tempfile.NamedTemporaryFile().name)
        _val = Path(tempfile.NamedTemporaryFile().name)

        train, val = sklearn.model_selection.train_test_split(
            self.train, test_size=0.1, random_state=23
        )

        with _train.open("w", encoding="utf-8") as file_:
            sentences = [
                "\n".join([f"{token.text} {token.label}" for token in sentence]).strip()
                for sentence in train
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
                for sentence in val
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

    def continued(self, name: str = "de-ner"):
        corpus = self._load_dataset()
        tagger = SequenceTagger.load(name)
        trainer = ModelTrainer(tagger, corpus)

        trainer.train(
            "continued", learning_rate=0.1, mini_batch_size=32, max_epochs=100
        )

        tagger = SequenceTagger.load("scratch/final-model.pt")
        pred = self._prediction(tagger)
        metric = evaluate(f"{name}-continued", self.test, pred)
        print(metric)
        return metric

    def scratch(self):
        corpus = self._load_dataset()
        tag_type = "ner"
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        embedding_types = [
            FlairEmbeddings("news-forward"),
            FlairEmbeddings("news-backward"),
        ]
        embeddings = StackedEmbeddings(embeddings=embedding_types)
        tagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=True,
        )

        trainer = ModelTrainer(tagger, corpus)

        trainer.train("scratch", learning_rate=0.1, mini_batch_size=32, max_epochs=100)

        tagger = SequenceTagger.load("scratch/final-model.pt")
        pred = self._prediction(tagger)
        metric = evaluate("scratch", self.test, pred)
        print(metric)
        return metric

    def vanilla(self, name: str = "de-ner"):
        tagger = SequenceTagger.load(name)
        pred = self._prediction(tagger)
        metric = evaluate(name, self.test, pred)
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
                    self.conll2custom.get(token.get_tag("ner").value, "O"),
                )
                for index, token in enumerate(sentence)
            ]
            preds.append(pred)
        return preds

    @property
    def custom2conll(self):
        return {
            "B-FIRST_NAME": "B-PER",
            "I-FIRST_NAME": "I-PER",
            "B-LAST_NAME": "B-PER",
            "I-LAST_NAME": "I-PER",
            "B-ORGANIZATION": "B-ORG",
            "I-ORGANIZATION": "I-ORG",
        }

    @property
    def conll2custom(self):
        return {
            "B-PER": "B-FIRST_NAME",
            "I-PER": "I-FIRST_NAME",
            "B-PER": "B-LAST_NAME",
            "I-PER": "I-LAST_NAME",
            "B-ORG": "B-ORGANIZATION",
            "I-ORG": "I-ORGANIZATION",
        }


class BERT:
    pass


class ConditionalRandomField:
    pass


class RuleBased:
    pass
