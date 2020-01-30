from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

from flair.datasets import ColumnCorpus
from flair.embeddings import PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


@dataclass
class Baseline:
    directory: Union[str, Path]
    train_file: str
    test_file: str
    dev_file: str
    column_format: Dict[int, str] = {0: "text", 1: "ner"}

    @property
    def corpus(self) -> ColumnCorpus:
        return ColumnCorpus(
            data_folder=self.directory,
            column_format=self.column_format,
            train_file=self.train_file,
            test_file=self.test_file,
            dev_file=self.dev_file,
        )

    def _train(
        self,
        directory: Union[str, Path],
        tagger: Optional[SequenceTagger] = None,
        hidden_size: int = 256,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        max_epochs: int = 100,
        use_crf: bool = True,
    ) -> SequenceTagger:
        tag_dictionary = self.corpus.make_tag_dictionary(tag_type="ner")
        if not tagger:
            tagger = SequenceTagger(
                hidden_size=hidden_size,
                embeddings=[PooledFlairEmbeddings("news-forward")],
                tag_dictionary=tag_dictionary,
                tag_type="ner",
                use_crf=use_crf,
            )
        trainer = ModelTrainer(tagger, self.corpus)
        trainer.train(
            directory,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=max_epochs,
        )
        model_path = Path(directory, "final-model.pt")
        return SequenceTagger.load(model_path)

    def _predict(self, tagger: SequenceTagger):
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

    def vanilla(self):
        model_path = _train_flair_model("vanilla", corpus)
        tagger = SequenceTagger.load(model_path)
        pred = self._prediction(tagger)
        metric = evaluate(name, self.test, pred)
        print(metric)
        return metric


'''

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
