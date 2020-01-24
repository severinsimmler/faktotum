import itertools
from collections import defaultdict
from typing import List

import pandas as pd
import numpy as np
import sklearn.utils
import statsmodels.stats.contingency_tables

from extract.corpus import Token


def bootstrap(dataset, modeling_function, evaluation_function, n_iterations=1000):
    n_size = int(len(dataset) * 0.50)
    stats = list()
    for _ in range(n_iterations):
        train = sklearn.utils.resample(dataset, n_samples=n_size)
        test = np.array([x for x in dataset if x.tolist() not in train.tolist()])
        model = modeling_function(train)
        pred = evaluation_function(model, test)
        metric = Metric(f"iter_{_}")
        score = {
            "precision": metric.precision(),
            "recall": metric.recall(),
            "f1": metric.f_score(),
            "accuracy": metric.accuracy(),
        }
        print(f"Iteration {_}")
        print(score)
        stats.append(score)
    print("Confidence intervals")
    print(f"Precision: {confidence_intervals([score['precision'] for score in stats])}")
    print(f"Recall: {confidence_intervals([score['recall'] for score in stats])}")
    print(f"F1: {confidence_intervals([score['f1'] for score in stats])}")
    print(f"Accuracy: {confidence_intervals([score['accuracy'] for score in stats])}")
    return stats


def confidence_intervals(stats):
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print(f"{alpha*100} confidence interval {lower*100} and {upper*100}")
    return alpha * 100, lower * 100, upper * 100


def compare_models(gold, pred1, pred2, alpha=0.05):
    table = get_contingency_table(gold, pred1, pred2)
    result = statsmodels.stats.contingency_tables.mcnemar(table)
    if result.pvalue > alpha:
        print("Same proportions of errors (fail to reject H0).")
        return False, result.pvalue
    else:
        print("Different proportions of errors (reject H0).")
        return True, result.pvalue


def get_contingency_table(gold, pred1, pred2):
    s1 = list()
    s2 = list()
    for _g, _p1, _p2 in zip(gold, pred1, pred2):
        for g, p1, p2 in zip(_g, _p1, _p2):
            if g.label != "O":
                if g.label == p1.label:
                    s1.append("agree")
                elif g.label != p1.label:
                    s1.append("disagree")

                if g.label == p2.label:
                    s2.append("agree")
                elif g.label != p2.label:
                    s2.append("disagree")
    table = pd.DataFrame({"model1": s1, "model2": s2})
    return pd.crosstab(table["model1"], table["model2"]).values


class Metric:
    def __init__(self, name):
        self.name = name
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if not class_name:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if not class_name:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if not class_name:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if not class_name:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fp(class_name)),
                4,
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fn(class_name)),
                4,
            )
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return round(
                2
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) + self.recall(class_name)),
                4,
            )
        return 0.0

    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name)
            > 0
        ):
            return round(
                (self.get_tp(class_name))
                / (
                    self.get_tp(class_name)
                    + self.get_fp(class_name)
                    + self.get_fn(class_name)
                ),
                4,
            )
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]
        if len(class_accuracy) > 0:
            return round(sum(class_accuracy) / len(class_accuracy), 4)
        return 0.0

    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        return all_classes

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.precision(), self.recall(), self.accuracy(), self.micro_avg_f_score()
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE".format(prefix)
        return "PRECISION\tRECALL\tACCURACY\tF-SCORE"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)


def evaluate(name: str, gold: List[List[Token]], pred: List[List[Token]]) -> Metric:
    metric = Metric(name)
    for sentence, sentence_ in zip(gold, pred):
        y_gold = [token for token in sentence if token.label != "O"]
        y_pred = [token for token in sentence_ if token.label != "O"]

        for token in y_pred:
            if token in y_gold:
                metric.add_tp(token.label)
            else:
                metric.add_fp(token.label)

        for token in y_gold:
            if token not in y_pred:
                metric.add_fn(token.label)
            else:
                metric.add_tn(token.label)
    return metric
