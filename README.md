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
    --data_dir extract/data/litbank \
    --model_type bert \
    --labels extract/data/litbank/labels.txt \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir /mnt/data/users/simmler/ner-models/gutenberg/bert-multi-litbank \
    --max_seq_length 128 \
    --num_train_epochs 1 \
    --per_gpu_train_batch_size 16 \
    --save_steps 754440 \
    --seed 23 --do_train \
    --do_eval \
    --do_predict && \
    poetry run bert-fine-tuning \
    --data_dir extract/data/droc \
    --model_type bert \
    --labels extract/data/droc/labels.txt \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir /mnt/data/users/simmler/ner-models/gutenberg/bert-multi-droc \
    --max_seq_length 128 \
    --num_train_epochs 2 \
    --per_gpu_train_batch_size 16 \
    --save_steps 5000744450 \
    --seed 23 --do_train \
    --do_eval \
    --do_predict && \
    poetry run bert-fine-tuning \
    --data_dir extract/data/droc \
    --model_type bert \
    --labels extract/data/droc/labels.txt \
    --model_name_or_path bert-base-german-dbmdz-cased \
    --output_dir /mnt/data/users/simmler/ner-models/gutenberg/bert-german-droc \
    --max_seq_length 128 \
    --num_train_epochs 2 \
    --per_gpu_train_batch_size 16 \
    --save_steps 11744450 \
    --seed 23 --do_train \
    --do_eval \
    --do_predict && \
    poetry run bert-fine-tuning \
    --data_dir extract/data/droc \
    --model_type bert \
    --labels extract/data/droc/labels.txt \
    --model_name_or_path /mnt/data/users/simmler/ner-models/gutenberg/bert-multi-litbank \
    --output_dir /mnt/data/users/simmler/ner-models/gutenberg/bert-german-litbank-continued-droc \
    --max_seq_length 128 \
    --num_train_epochs 2 \
    --per_gpu_train_batch_size 16 \
    --save_steps 117544440 \
    --seed 23 --do_train \
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
        [x] bert german gutenberg
        [x] bert multi gutenberg
        [ ] flair multi gutenberg
        [ ] flair historic gutenberg
    - presse
        [x] bert german presse
        [x] bert multi presse
        [ ] flair multi presse

- ner
    - gutenberg
        [ ] crf: droc (baseline)
        [x] flair: litbank
        [x] flair: droc
        [x] flair: droc plus litbank
        [x] bert multi: litbank
        [x] bert multi: droc
        [x] bert multi: droc continued litbank
        [x] bert german: droc
    - presse
        [ ] crf: presse (baseline)
        [x] flair: germeval
        [ ] flair: presse
        [ ] flair: presse continued germeval
        [x] bert multi: germeval
        [ ] bert multi: presse
        [ ] bert multi: presse continued germeval
        [x] bert german: germeval
        [ ] bert german: presse
        [ ] bert german: presse continued germeval
