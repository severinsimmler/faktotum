import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Dict, Optional, Union

import flair
import joblib
import sklearn_crfsuite
import torch
from flair.data import Sentence, MultiCorpus
from flair.datasets import ColumnCorpus
from flair.embeddings import PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from faktotum.evaluation import evaluate_labels


@dataclass
class Baseline:
    directory: Union[str, Path]
    train_file: str
    test_file: str
    dev_file: str

    def __post_init__(self):
        if "droc" in str(self.directory):
            module_folder = Path(__file__).parent
            features_file = Path(module_folder, "data", "kallimachos.json")
            self.kallimachos = json.loads(features_file.read_text(encoding="utf-8"))

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
        train_sents = list(self._parse_data(Path(self.directory, self.train_file)))
        test_sents = list(self._parse_data(Path(self.directory, self.test_file)))

        X_train = [self._sent2features(s) for s in train_sents]
        y_train = [self._sent2labels(s) for s in train_sents]

        X_test = [self._sent2features(s) for s in test_sents]
        y_test = [self._sent2labels(s) for s in test_sents]

        crf = sklearn_crfsuite.CRF(algorithm="l2sgd", verbose=True)
        crf.fit(X_train, y_train)

        y_pred = crf.predict(X_test)

        joblib.dump(crf, "crf-baseline.joblib")

        with Path("prediction.json").open("w", encoding="utf-8") as file_:
            json.dump({"gold": y_test, "pred": y_pred}, file_, indent=2)

        metric = evaluate_labels("crf-baseline", y_test, y_pred)
        print(metric)

        results = {
            "precision": metric.precision(),
            "recall": metric.recall(),
            "micro_f1": metric.micro_avg_f1(),
            "macro_f1": metric.macro_avg_f1(),
            "micro_accuracy": metric.micro_avg_accuracy(),
            "macro_accuracy": metric.macro_avg_accuracy(),
        }

        with Path("results.json").open("w", encoding="utf-8") as file_:
            json.dump(results, file_, indent=2)
        return metric

    def _word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][3]

        features = {
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "postag": postag,
            "postag[:2]": postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update(
                {
                    "-1:word.lower()": word1.lower(),
                    "-1:word.istitle()": word1.istitle(),
                    "-1:word.isupper()": word1.isupper(),
                    "-1:postag": postag1,
                    "-1:postag[:2]": postag1[:2],
                }
            )
        else:
            features["BOS"] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update(
                {
                    "+1:word.lower()": word1.lower(),
                    "+1:word.istitle()": word1.istitle(),
                    "+1:word.isupper()": word1.isupper(),
                    "+1:postag": postag1,
                    "+1:postag[:2]": postag1[:2],
                }
            )
        else:
            features["EOS"] = True

        features.update(self.kallimachos.get(word, dict()))
        return features

    def _sent2features(self, sent):
        return [self._word2features(sent, i) for i in range(len(sent))]

    def _sent2labels(self, sent):
        return [label for _, label, _, _ in sent]

    def _sent2tokens(self, sent):
        return [token for token, _, _, _ in sent]


@dataclass
class Flair:
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
            s = Sentence(
                " ".join([token for token, _ in sentence]), use_tokenizer=False
            )
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

    def multi_corpus(self, first_corpus: str = "germeval"):
        data_dir = Path(Path(self.directory).parent, first_corpus)
        first = self._load_corpus(data_dir)
        second = self._load_corpus()
        corpus = MultiCorpus([first, second])
        tagger = self._train("multi-corpus-model", corpus)
        metric = self._evaluate("multi-corpus-model", tagger)
        print(metric)
        return metric


@dataclass
class BERT:
    directory: Union[str, Path]
    train_file: str
    test_file: str
    dev_file: str

    def fine_tune(self, model_name_or_path: str, epochs=2):
        module = Path(__file__).resolve().parent
        script = Path(module, "vendor", "ner.py")
        command = [
            sys.executable,
            str(script),
            "--data_dir",
            str(self.directory),
            "--model_type",
            "bert",
            "--labels",
            str(Path(self.directory, "labels.txt")),
            "--model_name_or_path",
            model_name_or_path,
            "--output_dir",
            "bert-fine-tuned",
            "--max_seq_length",
            "128",
            "--num_train_epochs",
            str(epochs),
            "--per_gpu_train_batch_size",
            str(16),
            "--save_steps",
            "500",
            "--seed",
            "23",
            "--do_train",
            "--do_eval",
            "--do_predict",
        ]
        subprocess.check_call(command)