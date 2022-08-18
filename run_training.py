import argparse
import csv
import gzip
import logging
import math
import os
import random
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from sentence_transformers import (LoggingHandler, SentenceTransformer, losses,
                                   util)
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluatorForDev, BinaryClassificationEvaluatorForTest,
    EmbeddingSimilarityEvaluator)
from sentence_transformers.readers import InputExample
from utils import (DatasetEntry, build_sentence_transformer, get_eval_dataset,
                   softmax)


def download_sts_dataset(sts_dataset_path: str) -> None:
    """Download the STS dataset if it isn't already present."""
    if not os.path.exists(sts_dataset_path):
        util.http_get(
            "https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path
        )


def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    ds: List[DatasetEntry], dev_size: float = 0.1, seed: int = 42
) -> Dict[str, List[DatasetEntry]]:
    """Split a dataset into a train and dev set.

    The split is performed such that the distribution of labels is identical for the training and development set.

    :param ds: The dataset to split.
    :param dev_size: The relative size of the development set, in the range (0,1).
    :param seed: The seed used to initialize the random number generator.
    :return: A dictionary with keys "train" and "dev", whose values are the corresponding datasets.
    """
    train, dev = [], []
    rng = random.Random(seed)
    ds_grouped_by_label = defaultdict(list)
    for x in ds:
        ds_grouped_by_label[x.label].append(x)

    for label_list in ds_grouped_by_label.values():
        rng.shuffle(label_list)
        num_dev_examples = int(len(label_list) * dev_size)
        train += label_list[num_dev_examples:]
        dev += label_list[:num_dev_examples]

    return {"train": train, "dev": dev}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task", type=str, choices=["qqp", "mrpc", "stsb"])
    parser.add_argument(
        "--eval_task",
        type=str,
        choices=["qqp", "mrpc", "stsb", "paws", "paws_wiki"],
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument(
        "--input_train_file",
        type=str,
        required=True,
        help="The JSONL file that contains the DINO-generated dataset to train on.",
    )
    parser.add_argument(
        "--input_dev_file",
        type=str,
        required=True,
        help="The JSONL file that contains the DINO-generated dataset to validate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="./logs/{task}/{modelname}/",
        required=True,
        help="The output directory for storing the trained model and evaluation results.",
    )
    # Model and training parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="The pretrained Transformer language model to use.",
    )
    parser.add_argument(
        "--disc_model_name",
        type=str,
        default="bert-base-uncased",
        help="Discriminator에 사용된 backbone LM",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="The batch size used for training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="The number of epochs to train for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed used to initialize all random number generators.",
    )
    # Evaluation parameters
    parser.add_argument(
        "--sts_dataset_path",
        type=str,
        default="datasets/stsbenchmark.tsv.gz",
        help="The path to the STSb dataset. The STSb dataset is downloaded and saved at this path if it does not exist.",
    )
    parser.add_argument(
        "--classification_dataset_path",
        type=str,
        default="datasets/benchmarks/{}/",
        help="The path to the QQP or MRPC dataset.",
    )
    parser.add_argument(
        "--paws_dataset_path",
        type=str,
        default="datasets/benchmarks/paws/",
    )
    parser.add_argument(
        "--data_cls_to_reg",
        action="store_true",
    )
    parser.add_argument("--x1x2", action="store_true")
    parser.add_argument(
        "--disc_input_type",
        type=str,
        default="pair",
        choices=["pair", "single"],
    )
    parser.add_argument(
        "--filtering_ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--weight_ablation_strategy",
        default=None,
        choices=["uniform", "shuffle", "all_0.1"],
    )
    parser.add_argument(
        "--reweight_strategy",
        default="naive",
        choices=[
            "naive",
            "temp_0.9",
            "temp_0.7",
            "temp_0.5",
        ],
    )
    parser.add_argument("--machine_reweight", action="store_true")
    args = parser.parse_args()
    if args.eval_task is None:
        args.eval_task = args.task
    elif args.eval_task in ["paws", "paws_wiki"]:
        assert args.eval_only

    score_file_specification = args.task
    if args.x1x2:
        score_file_specification += "_x1x2"
        assert "-x1x2" in args.input_train_file and "-x1x2" in args.input_dev_file
    if args.data_cls_to_reg:
        score_file_specification += "-regression"
    args.leverage_score = args.filtering_ratio != 0.0 or args.machine_reweight

    if args.leverage_score:
        args.scoring_fname = (
            f"./data/discrimination/{score_file_specification}/{args.disc_input_type}/"
        )
        print(args.scoring_fname, os.listdir(args.scoring_fname))
        if "temp" in args.reweight_strategy:
            args.scoring_fname = os.path.join(
                args.scoring_fname,
                [e for e in os.listdir(args.scoring_fname) if "logits.txt" in e][0],
            )
        else:
            args.scoring_fname = os.path.join(
                args.scoring_fname,
                [e for e in os.listdir(args.scoring_fname) if "score.txt" in e][0],
            )
        print("Score: {}".format(args.scoring_fname))
        prefix = "_{}".format(args.disc_input_type)
        if args.filtering_ratio != 0.0:
            prefix += "_filter{}".format(args.filtering_ratio)
        elif args.machine_reweight:
            prefix += "_weight"
            if args.reweight_strategy != "naive":
                prefix += f"_{args.reweight_strategy}"
        if args.weight_ablation_strategy is not None:
            prefix += f"_{args.weight_ablation_strategy}"
        if score_file_specification != args.task:
            prefix += "_{}".format(score_file_specification)
        args.output_dir = os.path.join(args.output_dir, prefix)
    args.model_dir = os.path.join(args.output_dir, args.model_name)
    args.write_dir = os.path.join(args.output_dir, args.model_name)
    if args.task != args.eval_task:
        args.write_dir = args.write_dir.replace(
            args.model_name,
            args.model_name + "_{}".format(args.eval_task),
        )
    print("Write dir: {}".format(args.write_dir))
    # assert not os.path.exists(args.write_dir)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.write_dir, exist_ok=True)
    args.is_train_by_regression = True
    args.classification_dataset_path = args.classification_dataset_path.format(
        args.eval_task
    )
    return args


def read_score_fname(fname, is_logit=False):
    if is_logit:
        with open(fname, "r") as f:
            return [[float(s) for s in e.strip().split("\t")] for e in f.readlines()]
    else:
        with open(fname, "r") as f:
            return [float(e.strip()) for e in f.readlines()]


def filter_data_by_score(
    train_dataset,
    score_data,
    filtering_ratio,
):
    num_discard = int(len(train_dataset) * filtering_ratio)
    assert len(train_dataset) == len(score_data)
    sorted_indices = np.array(score_data).argsort()[::-1]

    selected_indices = sorted_indices[:-num_discard]
    return [e for i, e in enumerate(train_dataset) if i in selected_indices]


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    input_filename = os.path.basename(args.input_train_file)
    set_seed(args.seed)
    args.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Parameters: {args}")

    # We write all arguments to a file for better reproducibility.
    args_file = os.path.join(args.model_dir, f"args-{input_filename}.jsonl")
    with open(args_file, "w", encoding="utf8") as fh:
        fh.write(str(vars(args)))

    # If the STSb dataset does not exist, we download it.
    if args.task == "stsb":
        download_sts_dataset(args.sts_dataset_path)

    model = build_sentence_transformer(args.model_name)
    model_save_name = "_".join(
        [
            input_filename,
            args.model_name.replace("/", "-"),
        ]
    )
    model_save_path = os.path.join(args.model_dir, model_save_name)
    print(model_save_path)

    if not args.eval_only:
        print(model_save_path)
        try:
            assert not os.path.exists(model_save_path)
        except:
            if len(os.listdir(model_save_path)) != 0:
                raise ValueError

        train_dataset = DatasetEntry.read_list(
            args.input_train_file,
            keyname="question"
            if args.task == "qqp"
            else "sentence"
            if args.task == "mrpc"
            else None,
        )
        if args.leverage_score:
            score_data = read_score_fname(
                args.scoring_fname, "temp" in args.reweight_strategy
            )
            print(len(score_data), len(train_dataset))
            assert len(score_data) == len(train_dataset)
            if args.weight_ablation_strategy is not None:
                if args.weight_ablation_strategy == "uniform":
                    score_data = [
                        float(e) for e in np.random.uniform(0, 1, len(score_data))
                    ]
                elif args.weight_ablation_strategy == "shuffle":
                    random.shuffle(score_data)
                else:
                    raise ValueError
            if args.machine_reweight:
                if args.reweight_strategy == "naive":
                    pass
                elif "temp" in args.reweight_strategy:
                    temp = float(args.reweight_strategy.split("_")[1])
                    score_data = [softmax([s / temp for s in e])[1] for e in score_data]
                else:
                    raise ValueError(args.machine_reweight)
                for i, e in enumerate(score_data):
                    train_dataset[i].weight = score_data[i]
            else:  # Filter ablation
                assert "train.jsonl" in args.scoring_fname
                assert "train.jsonl" in args.input_train_file
                train_dataset = filter_data_by_score(
                    train_dataset,
                    score_data,
                    args.filtering_ratio,
                )
        dev_dataset = DatasetEntry.read_list(
            args.input_dev_file,
            keyname="question"
            if args.task == "qqp"
            else "sentence"
            if args.task == "mrpc"
            else None,
        )
        dataset = {"train": train_dataset, "dev": dev_dataset}

        train_samples = [
            InputExample(texts=[x.text_a, x.text_b], label=float(x.label))
            for x in dataset["train"]
        ]
        if args.machine_reweight:
            for i, w in enumerate(score_data):
                train_samples[i].weight = w

        dev_samples = [
            InputExample(texts=[x.text_a, x.text_b], label=float(x.label))
            for x in dataset["dev"]
        ]

        train_dataloader = DataLoader(
            train_samples, shuffle=True, batch_size=args.train_batch_size
        )

        warmup_steps = math.ceil(
            len(train_dataloader) * args.num_epochs * 0.1
        )  # 10% of train data for warm-up

        evaluation_frequency = int(len(train_dataloader) / 3)
        if args.machine_reweight:
            train_loss = losses.CosineSimilarityLoss(
                model=model, loss_fct=torch.nn.MSELoss(reduction="none")
            )
        else:
            train_loss = losses.CosineSimilarityLoss(model=model)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            dev_samples, name=f"{args.task}-dev"
        )
        # Train the model.
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=args.num_epochs,
            evaluation_steps=evaluation_frequency,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            batch_weight=args.machine_reweight,
        )

    # Evaluation
    if not args.skip_eval:
        model = SentenceTransformer(model_save_path)

        results = OrderedDict()
        # Evaluation step
        dev_samples, test_samples = get_eval_dataset(
            args.eval_task,
            args.sts_dataset_path
            if args.eval_task == "stsb"
            else args.paws_dataset_path
            if args.eval_task == "paws"
            else args.classification_dataset_path,
        )

        find_threshold_from_dev = False
        if args.eval_task in ["qqp", "mrpc", "paws"]:
            eval_class = BinaryClassificationEvaluatorForDev
            test_class = BinaryClassificationEvaluatorForTest
            find_threshold_from_dev = True
        else:
            eval_class = EmbeddingSimilarityEvaluator
            test_class = EmbeddingSimilarityEvaluator

        dev_evaluator = eval_class.from_input_examples(
            dev_samples, name=f"{args.eval_task}-dev"
        )
        test_evaluator = test_class.from_input_examples(
            test_samples, name=f"{args.eval_task}-test"
        )

        if find_threshold_from_dev:
            dev_result, acc_thresholds, f1_thresholds = dev_evaluator(
                model, output_path=args.write_dir
            )
            test_result = test_evaluator(
                model,
                output_path=args.write_dir,
                acc_threshold=acc_thresholds,
                f1_threshold=f1_thresholds,
            )
        else:
            dev_result = dev_evaluator(model, output_path=args.write_dir)
            test_result = test_evaluator(model, output_path=args.write_dir)
