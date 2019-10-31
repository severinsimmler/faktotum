# Extract information from unstructured text

## Getting started

```
$ poetry install
```

## Use cases

Train a fastText model on a custom corpus:

```
$ poetry run fasttext-training --model pretrained.fasttext --corpus novels --epochs 10
```

Tokenize and export a corpus to JSON:

```
$ poetry run tokenizer --corpus novels
```
