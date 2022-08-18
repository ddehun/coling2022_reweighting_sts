import argparse
import json
import os
import random

from datasets import load_dataset

# To make the list of {"text_a": "The sun is rising over the city.", "text_b": null, "label": "0.5"}

task_to_keys = {
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
}


def _save_split_data(dataset, raw_data_path):
    os.makedirs(os.path.dirname(raw_data_path.format(dataset, "train")), exist_ok=True)
    official_data = load_dataset("glue", dataset)
    assert all([split in official_data for split in ["train", "validation", "test"]])
    if official_data["test"][0]["label"] < 0:  # testset label not available
        official_data["validation"] = load_dataset(
            "glue",
            dataset,
            split=f"train[:5%]",
        )
        official_data["train"] = load_dataset(
            "glue",
            dataset,
            split=f"train[5%:]",
        )
        official_data["test"] = load_dataset(
            "glue",
            dataset,
            split="validation",
        )
    for setname in ["train", "validation", "test"]:
        with open(raw_data_path.format(dataset, setname), "w") as f:
            for el in official_data[setname]:
                f.write(json.dumps(el) + "\n")


def _read_json(fname):
    with open(fname, "r") as f:
        return [json.loads(e) for e in f.readlines()]


def _save_to_jsonl(data_dir, fname, output):
    os.makedirs(data_dir, exist_ok=True)
    with open(
        os.path.join(data_dir, fname),
        "w",
    ) as f:
        for e in output:
            f.write(json.dumps(e) + "\n")


def _make_first_sentence_datas(data, sent1_key, sent2_key, max_x1_num=15000):
    all_single_sentences = []
    for e in data:
        all_single_sentences.append(e[sent1_key])
        all_single_sentences.append(e[sent2_key])
    all_single_sentences = list(set(all_single_sentences))
    random.shuffle(all_single_sentences)
    all_single_sentences = all_single_sentences[:max_x1_num]
    outputs = [
        {"text_a": sent, "text_b": None, "label": -1} for sent in all_single_sentences
    ]
    return outputs


def main(args):
    for setname in ["train", "validation"]:
        raw_data_fname = args.raw_data_path.format(args.dataset, setname)
        if not os.path.exists(raw_data_fname):
            _save_split_data(args.dataset, args.raw_data_path)

        data = _read_json(raw_data_fname)
        sent1_key, sent2_key = task_to_keys[args.dataset]

        output = _make_first_sentence_datas(
            data, sent1_key, sent2_key, max_x1_num=args.max_x1_num
        )
        _save_to_jsonl(
            args.data_dir,
            args.output_fname_format.format(args.dataset, setname),
            output,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="stsb", choices=["stsb", "qqp", "mrpc"]
    )
    parser.add_argument("--max_x1_num", type=int, default=10000)
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="datasets/benchmarks/{}/{}.json",
    )
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument(
        "--output_fname_format", type=str, default="{}-x1-dataset.{}.jsonl"
    )
    args = parser.parse_args()

    main(args)
