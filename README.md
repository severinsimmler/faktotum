# faktotum information from unstructured text

## Getting started

Please use [Poetry](https://python-poetry.org/) to manage dependencies and a virtual environment. This is necessary (or at least recommended) to set up a deterministic build (e.g. to [reproduce numbers](#reproducing-numbers)).

```
$ poetry install
```

## Reproducing numbers

### Named entity recognition

Easy as:

```python
>>> from faktotum.ner import Baseline
>>> baseline = Baseline("droc",
...                     train_file="train.txt",
...                     dev_file="dev.txt",
...                     test_file="test.txt")
>>> baseline.from_scratch()
{
  'precision': 0.892,
  'recall': 0.7199,
  'micro_f1': 0.7968,
  'macro_f1': 0.7968,
  'micro_accuracy': 0.6622,
  'macro_accuracy': 0.6622
}
```

or:

```python
>>> from faktotum.ner import Flair
>>> flair = Flair("droc",
...               train_file="train.txt",
...               dev_file="dev.txt",
...               test_file="test.txt")
>>> flair.from_scratch()
{
  'precision': 0.8755,
  'recall': 0.6784,
  'micro_f1': 0.7644,
  'macro_f1': 0.7644,
  'micro_accuracy': 0.6187,
  'macro_accuracy': 0.6187
}
```

or:

```python
>>> from faktotum.ner import BERT
>>> bert = BERT("droc",
...             train_file="train.txt",
...             dev_file="dev.txt",
...             test_file="test.txt")
>>> bert.fine_tune("bert-base-german-dbmdz-cased")
{
  'precision': 0.921,
  'recall': 0.9544,
  'micro_f1': 0.9374,
  'macro_f1': 0.9374,
  'micro_accuracy': 0.8822,
  'macro_accuracy': 0.8822
}
```


## Todo

```
- language modeling
    - gutenberg
        [x] bert german gutenberg
        [x] bert multi gutenberg
    - presse
        [x] bert german presse
        [x] bert multi presse

- ner
    - gutenberg
        [x] crf: droc (baseline)
        [x] flair: litbank
        [x] flair: droc
        [x] flair: droc plus litbank
        [x] bert multi: litbank
        [x] bert multi: droc
        [x] bert multi: droc continued litbank
        [x] bert german: droc
        [x] bert fine-tuned multi: droc
        [x] bert fine-tuned german: droc
        [x] bert fine-tuned multi: litbank
        [x] bert fine-tuned multi: droc continued litbank
    - presse
        [x] crf: presse (baseline)
        [x] flair: germeval
        [x] flair: presse
        [x] flair: presse continued germeval
        [x] bert multi: germeval
        [x] bert multi: presse
        [x] bert multi: presse continued germeval
        [x] bert german: germeval
        [x] bert german: presse
        [x] bert german: presse continued germeval
        [x] bert fine-tuned multi: presse
        [x] bert fine-tuned german: presse
        [x] bert fine-tuned multi: germeval
        [x] bert fine-tuned multi: presse continued germeval
```
