# Extract information from unstructured text

## Getting started

Use [Poetry](https://python-poetry.org/) to manage dependencies and a virtual environment.


```
$ poetry install
```

## How to reproduce the numbers?

### Named entity recognition






```
poetry run bert-fine-tuning \
    --data_dir extract/data/ner/germeval \
    --model_type bert \
    --labels extract/data/ner/germeval/labels.txt \
    --model_name_or_path bert-base-german-dbmdz-cased \
    --output_dir german-germeval \
    --max_seq_length 128 \
    --num_train_epochs 1 \
    --per_gpu_train_batch_size 32 \
    --save_steps 750 \
    --seed 23 \
    --do_train \
    --do_eval \
    --do_predict
```



language-models
    presse
        - multi
        - german
    gutenberg
        - multi
        - german

ner-models
    baseline
        - trained on custom -> test on custom
        - trained on complete germeval/litbank -> test on custom
        - continued with custom on trained germeval/litbank -> test on custom
    conditional random field
        - classic features (POS-tags etc.)
    fine-tuned bert
        - multi
            - trained on custom -> test on custom
            - domain-adapted and trained on custom -> test on custom
        - german
            - trained on custom -> test on custom
            - domain-adapted and trained on custom -> test on custom
                -> the better: first germeval/litbank, then custom


TODO:
- language modeling
    - gutenberg
        [x] german gutenberg
        [x] multi gutenberg
    - presse
        [x] german presse
        [x] multi presse

- ner
    - gutenberg
        [ ] crf: droc (baseline)
        [ ] flair: litbank
        [ ] flair: droc
        [ ] flair: droc continued litbank
        [ ] bert multi: litbank
        [ ] bert multi: droc
        [ ] bert multi: droc continued litbank
        [ ] bert german: litbank
        [ ] bert german: droc
        [ ] bert german: droc continued litbank
    - presse
        [ ] crf: presse (baseline)
        [ ] flair: germeval
        [ ] flair: presse
        [ ] flair: presse continued germeval
        [ ] bert multi: germeval
        [ ] bert multi: presse
        [ ] bert multi: presse continued germeval
        [ ] bert german: germeval
        [ ] bert german: presse
        [ ] bert german: presse continued germeval
