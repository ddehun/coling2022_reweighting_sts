# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains some utility functions.
"""

import csv
import json
import os
import random
import gzip
from typing import Any, List, Optional

import numpy as np
import torch


import os
from sentence_transformers import util
import numpy as np

from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence


def get_use_encoder():
    import tensorflow_hub as hub

    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )
    return embed


class Dataset(object):
    def __init__(self, args, tokenizer, dataset, input_type):
        if input_type == "single":
            sequence = dataset["text_b"]
        else:
            sequence = [
                e1 + "[SEP]" + e2
                for e1, e2 in zip(dataset["text_a"], dataset["text_b"])
            ]
        tokenized_sequence = tokenizer(
            sequence, max_length=args.max_length, truncation=True
        )
        self.input_ids = tokenized_sequence["input_ids"]
        self.labels = dataset["label"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index]


def collate_fn(samples):
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)
    # sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1]
    len_inputs = range(len(input_ids))

    input_ids = pad_sequence(
        [torch.tensor(input_ids[index]) for index in len_inputs],
        batch_first=True,
    )
    attention_mask = torch.tensor(
        [
            [1] * len(input_ids[index])
            + [0] * (max_len - len(input_ids[index]))
            for index in len_inputs
        ]
    )
    token_type_ids = torch.tensor(
        [[0] * len(input_ids[index]) for index in len_inputs]
    )
    position_ids = torch.tensor(
        [list(range(len(input_ids[index]))) for index in len_inputs]
    )
    labels = torch.tensor(np.stack(labels, axis=0)[len_inputs])

    return input_ids, attention_mask, token_type_ids, position_ids, labels


def build_sentence_transformer(
    model_name: str, mean_pool=True, cls_pool=False, max_pool=False
):
    from sentence_transformers import SentenceTransformer, losses, models
    from sentence_transformers.readers import InputExample

    """Build the Sentence Transformer model."""
    assert sum([mean_pool, cls_pool, max_pool]) == 1
    try:
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=mean_pool,
            pooling_mode_cls_token=cls_pool,
            pooling_mode_max_tokens=max_pool,
        )
        return SentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )
    except OSError:
        return SentenceTransformer(model_name)


def get_eval_dataset(task, dataset_path):
    from sentence_transformers import SentenceTransformer, losses, models
    from sentence_transformers.readers import InputExample

    if task == "stsb":
        if not os.path.exists(dataset_path):
            util.http_get(
                "https://sbert.net/datasets/stsbenchmark.tsv.gz",
                dataset_path,
            )
        train_samples = []
        dev_samples = []
        test_samples = []
        with gzip.open(dataset_path, "rt", encoding="utf8") as fIn:
            reader = csv.DictReader(
                fIn, delimiter="\t", quoting=csv.QUOTE_NONE
            )
            for row in reader:
                score = (
                    float(row["score"]) / 5.0
                )  # Normalize score to range 0 ... 1
                inp_example = InputExample(
                    texts=[row["sentence1"], row["sentence2"]], label=score
                )
                if row["split"] == "dev":
                    dev_samples.append(inp_example)
                elif row["split"] == "test":
                    test_samples.append(inp_example)
                elif row["split"] == "train":
                    train_samples.append(inp_example)
                else:
                    raise ValueError
    elif task in ["qqp", "mrpc"]:
        key = "question" if task == "qqp" else "sentence"
        with open(dataset_path + "validation.json", "r") as f:
            dev_samples = [
                InputExample(
                    texts=[
                        json.loads(e)[key + str(1)],
                        json.loads(e)[key + str(2)],
                    ],
                    label=json.loads(e)["label"],
                )
                for e in f.readlines()
            ]
        with open(dataset_path + "test.json", "r") as f:
            test_samples = [
                InputExample(
                    texts=[
                        json.loads(e)[key + str(1)],
                        json.loads(e)[key + str(2)],
                    ],
                    label=json.loads(e)["label"],
                )
                for e in f.readlines()
            ]
    elif task in ["paws", "paws_wiki"]:

        key = "sentence"
        with open(dataset_path + "validation.json", "r") as f:
            dev_samples = [
                InputExample(
                    texts=[
                        json.loads(e)[key + str(1)],
                        json.loads(e)[key + str(2)],
                    ],
                    label=json.loads(e)["label"],
                )
                for e in f.readlines()
            ]

        key = "sentence"
        with open(dataset_path + "test.json", "r") as f:
            test_samples = [
                InputExample(
                    texts=[
                        json.loads(e)[key + str(1)],
                        json.loads(e)[key + str(2)],
                    ],
                    label=json.loads(e)["label"],
                )
                for e in f.readlines()
            ]
    else:
        raise ValueError
    return dev_samples, test_samples


def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_inputs(input_file: str, input_file_type: str) -> List[str]:
    """
    Read a list of input texts from a text file.
    :param input_file: the path to the input file
    :param input_file_type: the file type, one of 'plain', 'jsonl' and 'stsb':
        <ul>
            <li>'plain': a plain text file where each line corresponds to one input</li>
            <li>'jsonl': a jsonl file, where each line is one json object and input texts are stored in the field 'text_a'</li>
            <li>'stsb': a tsv file, formatted like the official STS benchmark</li>
        </ul>
    :return: the list of extracted input texts
    """
    valid_types = ["plain", "jsonl", "stsb"]
    assert (
        input_file_type in valid_types
    ), f"Invalid input file type: '{input_file_type}'. Valid types: {valid_types}"

    if input_file_type == "plain":
        return read_plaintext_inputs(input_file)
    elif input_file_type == "jsonl":
        return read_jsonl_inputs(input_file)
    elif input_file_type == "stsb":
        return read_sts_inputs(input_file)


def read_plaintext_inputs(path: str) -> List[str]:
    """Read input texts from a plain text file where each line corresponds to one input"""
    with open(path, "r", encoding="utf8") as fh:
        inputs = fh.read().splitlines()
    print(f"Done loading {len(inputs)} inputs from file '{path}'")
    return inputs


def read_jsonl_inputs(path: str) -> List[str]:
    """Read input texts from a jsonl file, where each line is one json object and input texts are stored in the field 'text_a'"""
    ds_entries = DatasetEntry.read_list(path)
    print(f"Done loading {len(ds_entries)} inputs from file '{path}'")
    return [entry.text_a for entry in ds_entries]


def read_sts_inputs(path: str) -> List[str]:
    """Read input texts from a tsv file, formatted like the official STS benchmark"""
    inputs = []
    with open(path, "r", encoding="utf8") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            try:
                sent_a, sent_b = row[5], row[6]
                inputs.append(sent_a)
                inputs.append(sent_b)
            except IndexError:
                print(f"Cannot parse line {row}")
    print(f"Done loading {len(inputs)} inputs from file '{path}'")
    return inputs


class DatasetEntry:
    """This class represents a dataset entry for text (pair) classification"""

    def __init__(self, text_a: str, text_b: Optional[str], label: Any):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        if self.text_b is not None:
            return f'DatasetEntry(text_a="{self.text_a}", text_b="{self.text_b}", label={self.label})'
        else:
            return f'DatasetEntry(text_a="{self.text_a}", label={self.label})'

    def __key(self):
        return self.text_a, self.text_b, self.label

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, DatasetEntry):
            return self.__key() == other.__key()
        return False

    @staticmethod
    def save_list(
        entries: List["DatasetEntry"], path: str, key1=None, key2=None
    ):
        with open(path, "w", encoding="utf8") as fh:
            for entry in entries:
                item = entry.__dict__
                if key1 is not None and key2 is not None:
                    item[key1] = item["text_a"]
                    item[key2] = item["text_b"]
                    del item["text_a"]
                    del item["text_b"]
                fh.write(f"{json.dumps(item)}\n")

    @staticmethod
    def read_list(path: str, keyname: str = None) -> List["DatasetEntry"]:
        # keyname: json에 저장된 key가 text_a, text_b가 아닐 경우
        pairs = []
        if "stsb" in path:
            path = path.replace("stsb", "sts")
        with open(path, "r", encoding="utf8") as fh:
            for line in fh:
                item = json.loads(line)
                if "text_a" not in item and keyname is not None:
                    item["text_a"] = item[keyname + str(1)]
                    item["text_b"] = item[keyname + str(2)]
                    del item[keyname + str(1)]
                    del item[keyname + str(2)]
                    del item["idx"]
                pairs.append(DatasetEntry(**item))
        return pairs


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
