# Extract information from unstructured text

## Getting started

```
$ poetry install
```

## Fine-tuning

```
poetry run bert-fine-tuning \
    --data_dir extract/data/germeval
    --model_type bert \
    --labels extract/data/germeval/labels.txt \
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
