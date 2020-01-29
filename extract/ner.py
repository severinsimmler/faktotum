import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import sklearn.model_selection
import torch
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from extract.corpus import Token
from extract.evaluation import evaluate


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    labels.append("O")
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index), words=words, labels=labels
                )
            )
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_ids.extend(
                [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
            )

        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
            )
        )
    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return [
            "O",
            "B-MISC",
            "I-MISC",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
        ]


Dataset = List[List[Token]]


@dataclass
class Baseline:
    train: Dataset
    val: Dataset
    test: Dataset

    def __post_init__(self):
        self._translate_labels(self.train)
        self._translate_labels(self.val)
        self._translate_labels(self.test)

    def _translate_labels(self, data):
        for sentence in data:
            previous = "[START]"
            for token in sentence:
                if (
                    previous == "I-FIRST_NAME" or previous == "B-FIRST_NAME"
                ) and token.label == "B-LAST_NAME":
                    label = "I-PER"
                else:
                    label = self.custom2germeval.get(token.label, "O")
                previous = token.label
                token.label = label

    def _load_custom_dataset(self):
        _train = Path(tempfile.NamedTemporaryFile().name)
        _test = Path(tempfile.NamedTemporaryFile().name)
        _val = Path(tempfile.NamedTemporaryFile().name)

        with _train.open("w", encoding="utf-8") as file_:
            sentences = [
                "\n".join([f"{token.text} {token.label}" for token in sentence]).strip()
                for sentence in self.train
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
                for sentence in self.val
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

    def _load_germeval_dataset(self):
        current_folder = Path(__file__).parent
        data_folder = Path(current_folder, "data", "germeval")
        columns = {1: "text", 2: "ner"}
        return ColumnCorpus(
            data_folder,
            columns,
            train_file="train.tsv",
            test_file="test.tsv",
            dev_file="dev.tsv",
        )

    def _train_flair_model(self, name, corpus, tagger=None):
        tag_dictionary = corpus.make_tag_dictionary(tag_type="ner")
        if not tagger:
            tagger = SequenceTagger(
                hidden_size=256,
                embeddings=[PooledFlairEmbeddings("news-forward")],
                tag_dictionary=tag_dictionary,
                tag_type=tag_type,
                use_crf=True,
            )
        trainer = ModelTrainer(tagger, corpus)
        trainer.train(name, learning_rate=0.1, mini_batch_size=32, max_epochs=50)
        return Path(name, "final-model.pt")

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

    @property
    def custom2germeval(self):
        return {
            "B-FIRST_NAME": "B-PER",
            "I-FIRST_NAME": "I-PER",
            "B-LAST_NAME": "B-PER",
            "I-LAST_NAME": "I-PER",
            "B-ORGANIZATION": "B-ORG",
            "I-ORGANIZATION": "I-ORG",
        }

    @property
    def germeval2custom(self):
        return {
            "B-PER": "B-PER",
            "I-PER": "I-PER",
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
        }


class BERT:
    train: Dataset
    val: Dataset
    test: Dataset
    labels: List[str] = ["B-PER", "I-PER", "B-ORG", "I-ORG"]

    data_dir: str
    model_type: str = "bert"
    model_name_or_path: str = "bert"
    output_dir: str = ""
    labels: str = ""
    max_seq_length: str = 128
    do_train: bool = True
    do_predict: bool = True
    evaluate_during_training: bool = False
    per_gpu_train_batch_size: int = 8
    per_gpu_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 50
    save_steps: int = 50
    eval_all_checkpoints: bool = False
    no_cuda: bool = False
    overwrite_output_dir: bool = False
    overwrite_cache: bool = False
    seed: int = 23
    fp16: bool = False
    fp16_opt_level: str = "01"
    local_rank: int = -1
    server_ip: str = ""
    server_port: str = ""
    device: str = torch.device("cuda")

    def __post_init__(self):
        self._translate_labels(self.train)
        self._translate_labels(self.val)
        self._translate_labels(self.test)

    @property
    def labels(self):
        l = set(self.germeval2custom)
        l.add("O")
        return l

    def _get_germeval_labels(self):
        pass

    def fine_tune(self, name: str, continued=False):
        self._set_seed()

        num_labels = len(self.labels)
        pad_token_label_id = CrossEntropyLoss().ignore_index

        config = BertConfig.from_pretrained(
            self.model_name_or_path, num_labels=num_labels
        )
        tokenizer = BertTokenizer.from_pretrained(
            self.model_name_or_path, do_lower_case=False
        )
        model = BertForTokenClassification.from_pretrained(
            self.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_name_or_path),
            config=config,
        )

        model.to(self.device)

        train_dataset = self._load_and_cache_examples(
            tokenizer, labels, pad_token_label_id, mode="train"
        )
        global_step, tr_loss = self._train(
            train_dataset, model, tokenizer, labels, pad_token_label_id
        )
        logging.info(f" global_step = {global_step}, average loss = {tr_loss}")

        if self.local_rank == -1:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            logging.info(f"Saving model checkpoint to {self.output_dir}")
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)

            torch.save(args, os.path.join(self.output_dir, "training_args.bin"))

        tokenizer = BertTokenizer.from_pretrained(self.output_dir, do_lower_case=False)
        model = BertForTokenClassification.from_pretrained(self.output_dir)
        model.to(self.device)
        result, predictions = self._evaluate(
            model, tokenizer, labels, pad_token_label_id, mode="test"
        )
        return results

    def _set_seed(self):
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
        torch.cuda.manual_seed_all(23)

    def _train(self, train_dataset, model, tokenizer, labels, pad_token_label_id):
        tb_writer = SummaryWriter()

        self.train_batch_size = self.per_gpu_train_batch_size
        train_sampler = (
            RandomSampler(train_dataset)
            if self.local_rank == -1
            else DistributedSampler(train_dataset)
        )
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=self.train_batch_size
        )

        if self.max_steps > 0:
            t_total = self.max_steps
            self.num_train_epochs = (
                self.max_steps
                // (len(train_dataloader) // self.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.gradient_accumulation_steps
                * self.num_train_epochs
            )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
        )

        if os.path.isfile(
            os.path.join(self.model_name_or_path, "optimizer.pt")
        ) and os.path.isfile(os.path.join(self.model_name_or_path, "scheduler.pt")):
            optimizer.load_state_dict(
                torch.load(os.path.join(self.model_name_or_path, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(self.model_name_or_path, "scheduler.pt"))
            )

        if self.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {self.num_train_epochs}")
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if os.path.exists(self.model_name_or_path):
            global_step = int(self.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (
                len(train_dataloader) // self.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // self.gradient_accumulation_steps
            )

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained,
            int(self.num_train_epochs),
            desc="Epoch",
            disable=self.local_rank not in [-1, 0],
        )
        self._set_seed()
        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
                disable=self.local_rank not in [-1, 0],
            )
            for step, batch in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                inputs["token_type_ids"] = (
                    batch[2] if self.model_type in ["bert", "xlnet"] else None
                )

                outputs = model(**inputs)
                loss = outputs[0]

                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )

                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    if (
                        self.local_rank in [-1, 0]
                        and self.logging_steps > 0
                        and global_step % self.logging_steps == 0
                    ):
                        if self.local_rank == -1 and self.evaluate_during_training:
                            results, _ = evaluate(
                                self,
                                model,
                                tokenizer,
                                labels,
                                pad_token_label_id,
                                mode="dev",
                            )
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, global_step
                                )
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / self.logging_steps,
                            global_step,
                        )
                        logging_loss = tr_loss

                    if (
                        self.local_rank in [-1, 0]
                        and self.save_steps > 0
                        and global_step % self.save_steps == 0
                    ):
                        output_dir = os.path.join(
                            self.output_dir, "checkpoint-{}".format(global_step)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(self, os.path.join(output_dir, "training_args.bin"))
                        logging.info(f"Saving model checkpoint to {output_dir}")

                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logging.info(
                            f"Saving optimizer and scheduler states to {output_dir}"
                        )

                if self.max_steps > 0 and global_step > self.max_steps:
                    epoch_iterator.close()
                    break
            if self.max_steps > 0 and global_step > self.max_steps:
                train_iterator.close()
                break

        if self.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step

    def _evaluate(self, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
        eval_dataset = self._load_and_cache_examples(
            tokenizer, labels, pad_token_label_id, mode=mode
        )

        self.eval_batch_size = self.per_gpu_eval_batch_size
        eval_sampler = (
            SequentialSampler(eval_dataset)
            if self.local_rank == -1
            else DistributedSampler(eval_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.eval_batch_size
        )

        logging.info("***** Running evaluation *****")
        logging.info(f"  Num examples = {len(eval_dataset)}")
        logging.info(f"  Batch size = {self.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                inputs["token_type_ids"] = (
                    batch[2] if self.model_type in ["bert", "xlnet"] else None
                )
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if self.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    def _load_and_cache_examples(self, tokenizer, labels, pad_token_label_id, mode):
        if self.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()

        cached_features_file = os.path.join(
            self.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(self.max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file) and not self.overwrite_cache:
            logging.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logging.info("Creating features from dataset file at %s", self.data_dir)
            examples = read_examples_from_file(self.data_dir, mode)
            features = convert_examples_to_features(
                examples,
                labels,
                self.max_seq_length,
                tokenizer,
                cls_token_at_end=bool(self.model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(self.model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(self.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
                pad_token_label_id=pad_token_label_id,
            )
            if self.local_rank in [-1, 0]:
                logging.info(
                    "Saving features into cached file %s", cached_features_file
                )
                torch.save(features, cached_features_file)

        if self.local_rank == 0 and not evaluate:
            torch.distributed.barrier()

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        return dataset

    @property
    def custom2germeval(self):
        return {
            "B-FIRST_NAME": "B-PER",
            "I-FIRST_NAME": "I-PER",
            "B-LAST_NAME": "B-PER",
            "I-LAST_NAME": "I-PER",
            "B-ORGANIZATION": "B-ORG",
            "I-ORGANIZATION": "I-ORG",
        }

    @property
    def germeval2custom(self):
        return {
            "B-PER": "B-PER",
            "I-PER": "I-PER",
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
        }


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
