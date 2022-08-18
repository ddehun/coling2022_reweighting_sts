import argparse
import os

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluatorForDev, BinaryClassificationEvaluatorForTest,
    EmbeddingSimilarityEvaluator)
from utils import build_sentence_transformer, get_eval_dataset

modellist = {
    "glove-avg": "average_word_embeddings_glove.840B.300d",  # Avg. Glove
    "bertbase-nli-mean": "nli-bert-base",  # S-BERT (base)
    "robertabase-nli": "nli-roberta-base",  # S-RoBERTa (base)
    "bertbase-cls": ["bert-base-uncased", False, True, False],
    "bertbase-mean": ["bert-base-uncased", True, False, False],
    "robertabase-mean": ["roberta-base", True, False, False],
    "infersent": None,
    "use": None,
}


def load_sentence_transformer_model(modelname):
    modelinfo = modellist[modelname]
    if modelinfo is None:
        raise NotImplementedError
    if isinstance(modelinfo, str):
        return SentenceTransformer(modelinfo)
    if isinstance(modelinfo, list):
        return build_sentence_transformer(*modelinfo)


def main(args):
    model = load_sentence_transformer_model(args.model_name)
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

    find_threshold_from_dev = False
    if args.task in ["qqp", "mrpc", "paws"]:
        dev_class = BinaryClassificationEvaluatorForDev
        test_class = BinaryClassificationEvaluatorForTest
        find_threshold_from_dev = True
    else:
        dev_class = EmbeddingSimilarityEvaluator
        test_class = EmbeddingSimilarityEvaluator

    dev_evaluator = dev_class.from_input_examples(dev_samples, name=f"{args.task}-dev")
    test_evaluator = test_class.from_input_examples(
        test_samples, name=f"{args.task}-test"
    )

    if find_threshold_from_dev:
        _, acc_thresholds, f1_thresholds = dev_evaluator(
            model, output_path=args.write_dir
        )
        _ = test_evaluator(
            model,
            output_path=args.write_dir,
            acc_threshold=acc_thresholds,
            f1_threshold=f1_thresholds,
        )
    else:
        _ = dev_evaluator(model, output_path=args.write_dir)
        _ = test_evaluator(model, output_path=args.write_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task",
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
        default="bertbase-nli-mean",
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
        default="./datasets/benchmarks/{}/",
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
