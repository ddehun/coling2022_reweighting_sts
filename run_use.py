import argparse
import csv
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import get_eval_dataset, get_use_encoder


def return_acc(scores, labels, best_threshold: float):
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    rows = list(zip(scores, labels))
    predict_results = []

    for i in range(len(rows)):
        score, label = rows[i]

        if score > best_threshold:
            predict = 1
        else:
            predict = 0

        predict_results.append(predict)

    predict_results = np.asarray(predict_results)

    return np.mean(labels == predict_results)


def return_f1(scores, labels, best_threshold: float):
    from sklearn.metrics import f1_score, precision_score, recall_score

    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    rows = list(zip(scores, labels))
    predict_results = []

    for i in range(len(rows)):
        score, label = rows[i]

        if score > best_threshold:
            predict = 1
        else:
            predict = 0

        predict_results.append(predict)

    f1 = f1_score(y_true=labels, y_pred=np.asarray(predict_results))
    precision = precision_score(y_true=labels, y_pred=np.asarray(predict_results))
    recall = recall_score(y_true=labels, y_pred=np.asarray(predict_results))

    return f1, precision, recall


def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)
    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    max_acc = 0
    best_threshold = -1

    positive_so_far = 0
    remaining_negatives = sum(labels == 0)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        if label == 1:
            positive_so_far += 1
        else:
            remaining_negatives -= 1

        acc = (positive_so_far + remaining_negatives) / len(labels)
        if acc > max_acc:
            max_acc = acc
            best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return max_acc, best_threshold


def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    best_f1 = best_precision = best_recall = 0
    threshold = 0
    nextract = 0
    ncorrect = 0
    total_num_duplicates = sum(labels)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return best_f1, best_precision, best_recall, threshold


def main(args):
    model = get_use_encoder()
    dev_samples, test_samples = get_eval_dataset(
        args.task,
        args.sts_dataset_path
        if args.task == "stsb"
        else args.paws_dataset_path
        if args.task == "paws"
        else args.paws_wiki_dataset_path
        if args.task == "paws_wiki"
        else args.classification_dataset_path,
    )

    dev_samples = dev_samples
    test_samples = test_samples
    is_regression = args.task not in ["qqp", "mrpc", "paws"]

    if is_regression:
        test_scores = []
        test_labels = []
        for e in tqdm(test_samples):
            test_labels.append(e.label)
            encoded = model(e.texts)
            score = float(cosine_similarity([encoded[0]], [encoded[1]]))
            test_scores.append(score)
        eval_pearson_cosine, _ = pearsonr(test_labels, test_scores)
        eval_spearman_cosine, _ = spearmanr(test_labels, test_scores)
        write_thing = [
            eval_pearson_cosine,
            eval_spearman_cosine,
            eval_pearson_cosine,
            eval_spearman_cosine,
        ]
    else:
        dev_scores = []
        dev_labels = []
        test_scores = []
        test_labels = []
        for e in tqdm(dev_samples):
            dev_labels.append(e.label)
            encoded = model(e.texts)
            score = float(cosine_similarity([encoded[0]], [encoded[1]]))
            dev_scores.append(score)
        for e in tqdm(test_samples):
            test_labels.append(e.label)
            encoded = model(e.texts)
            score = float(cosine_similarity([encoded[0]], [encoded[1]]))
            test_scores.append(score)

        acc, acc_threshold = find_best_acc_and_threshold(
            dev_scores, np.array(dev_labels), True
        )
        (
            f1,
            precision,
            recall,
            f1_threshold,
        ) = find_best_f1_and_threshold(dev_scores, np.array(dev_labels), True)
        acc = return_acc(test_scores, test_labels, acc_threshold)
        (f1, precision, recall) = return_f1(test_scores, test_labels, f1_threshold)
        write_thing = [acc, acc_threshold, f1, f1_threshold]
    print(write_thing)
    os.makedirs(args.write_dir, exist_ok=True)
    with open(
        os.path.join(args.write_dir, f"{args.model_name}_result.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.write_row(write_thing)
        writer.write_row(write_thing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task",
        default="qqp",
        type=str,
        choices=["stsb", "qqp", "mrpc", "paws", "paws_wiki"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_unsup/{}/",
        help="The output directory for storing the trained model and evaluation results.",
    )

    # Model and training parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="use",
        help="The pretrained Transformer language model to use.",
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
        help="The path to the QQP or MRPC dataset.",
    )
    parser.add_argument(
        "--paws_dataset_path",
        type=str,
    )
    parser.add_argument(
        "--paws_wiki_dataset_path",
        type=str,
    )
    args = parser.parse_args()
    args.write_dir = os.path.join(args.output_dir.format(args.task), args.model_name)
    os.makedirs(args.write_dir, exist_ok=True)
    if args.task in ["qqp", "mrpc", "paws"]:
        args.classification_dataset_path = args.classification_dataset_path.format(
            args.task
        )

    main(args)
