import argparse
import json
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup,
                          set_seed)

from utils import Dataset, collate_fn


def compute_acc(predictions, target_labels):
    return (np.array(predictions) == np.array(target_labels)).mean()


# Argment parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="stsb",
        choices=[
            "qqp",
            "mrpc",
            "stsb",
        ],
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument(
        "--data_cls_to_reg",
        action="store_true",
    )
    parser.add_argument("--do_train", type=str, default="t")
    parser.add_argument("--do_test", type=str, default="t")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--x1x2", action="store_true", help="if both x1 and x2 are generated."
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="single",
        choices=["single", "pair"],
        help="Input type for a source discrimination task. single: x2, pair: x1x2",
    )

    args = parser.parse_args()

    config = defaultdict()
    for arg, value in args._get_kwargs():
        config[arg] = value
    args.human_data_fname = f"datasets/benchmarks/{args.dataset}/" + "{}.json"

    # Where is machine-written data?
    if args.x1x2:
        model_data_specification = f"{args.dataset}_x1x2"
        args.model_data_fname = "./data/sts-x1x2-dataset.postprocessed.{}.jsonl"
    else:
        model_data_specification = args.dataset
        if args.data_cls_to_reg:
            model_data_specification += "-regression"
        args.model_data_fname = (
            f"./data/{model_data_specification}-dataset.postprocessed." + "{}.jsonl"
        )
        if args.seed != 42:
            args.model_data_fname = args.model_data_fname.replace(
                ".jsonl", f"-seed{args.seed}.jsonl"
            )
    if args.dataset == "stsb":
        args.model_data_fname = args.model_data_fname.replace("stsb", "sts")

    # Where to save scores
    output_data_specification = model_data_specification
    if args.model_name != "bert-base-uncased":
        output_data_specification += f"-{args.model_name}"

    args.exp_dir = "data/discrimination/{}/{}/".format(
        output_data_specification, args.input_type
    )

    # Scoring한 파일 이름 및 각 샘플들에 대한 score
    args.output_fname = os.path.join(
        args.exp_dir,
        "{}.score.txt".format(os.path.basename(args.model_data_fname.format("train"))),
    )
    if args.save_logits:
        args.output_fname = args.output_fname.replace("score.txt", "logits.txt")
    assert not os.path.exists(args.output_fname)
    os.makedirs(args.exp_dir, exist_ok=True)
    return args


def str_to_boolean(str):
    if str == "t" or str == "T" or str == "True":
        return True
    elif str == "f" or str == "F" or str == "False":
        return False
    else:
        raise ValueError("String must be t or T for True and f or F for False")


keymap = {
    "stsb": ["sentence1", "sentence2"],
    "qqp": ["question1", "question2"],
    "mrpc": ["sentence1", "sentence2"],
}


def make_data_for_scoring_by_discrimination(model_fname, input_type):
    with open(model_fname, "r") as f:
        model_data = [json.loads(e) for e in f.readlines()]
    remain_keylist = (
        ["text_a", "text_b"]
        if input_type == "pair"
        else ["text_b"]
        if input_type == "single"
        else None
    )
    remain_keylist.append("label")
    output = {k: [] for k in remain_keylist}
    for idx, el in enumerate(model_data):
        del_keylist = []
        for k, v in el.items():
            if k in remain_keylist:
                output[k].append(v)
    return output


def make_data_for_discrimination(human_fname, model_fname, keypair, input_type):
    with open(human_fname, "r") as f:
        human_data = [json.loads(e) for e in f.readlines()]
        for idx, el in enumerate(human_data):
            el["text_a"] = el[keypair[0]]
            el["text_b"] = el[keypair[1]]
            del el[keypair[0]]
            del el[keypair[1]]
            human_data[idx] = el
    with open(model_fname, "r") as f:
        model_data = [json.loads(e) for e in f.readlines()]

    human_string_list = []
    remain_keylist = (
        ["text_a", "text_b"]
        if input_type == "pair"
        else ["text_b"]
        if input_type == "single"
        else None
    )
    print(remain_keylist)
    for idx, el in enumerate(human_data):
        del_keylist = []
        for k, v in el.items():
            if k not in remain_keylist:
                del_keylist.append(k)
        del_keylist = list(set(del_keylist))
        for del_k in del_keylist:
            del el[del_k]
        el["label"] = 1  # 1 for human
        human_data[idx] = el
        human_string_list.append(
            " ".join([el[remain_key] for remain_key in remain_keylist])
        )
    print(human_data[0])
    model_identical_w_human_indices = []
    from tqdm import tqdm

    del_keylist = []
    for k, v in model_data[0].items():
        if k not in remain_keylist:
            del_keylist.append(k)
    del_keylist = list(set(del_keylist))

    for idx, el in enumerate(tqdm(model_data)):
        del_keylist = []
        for k, v in el.items():
            if k not in remain_keylist:
                del_keylist.append(k)
        del_keylist = list(set(del_keylist))
        for del_k in del_keylist:
            del el[del_k]
        el["label"] = 0  # 0 for machine
        modelstring = " ".join([el[remain_key] for remain_key in remain_keylist])
        if modelstring in human_string_list:
            model_identical_w_human_indices.append(idx)
        else:
            model_data[idx] = el
    model_data = [
        el
        for idx, el in enumerate(model_data)
        if idx not in model_identical_w_human_indices
    ]
    print(model_data[0])

    minimum_length = min(len(model_data), len(human_data))
    random.shuffle(model_data)
    model_data = model_data[:minimum_length]
    human_data = human_data[:minimum_length]
    output = {}
    for k in remain_keylist:
        output[k] = []
    output["label"] = []
    for item in model_data + human_data:
        for k, v in item.items():
            output[k].append(v)
    pos_ratio = round(100 * sum(output["label"]) / len(output["label"]), 2)
    print("Label ratio: {} and {}%".format(100 - pos_ratio, pos_ratio))
    return output


def main():
    args = get_args()
    print(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, "board"))
    args.num_labels = 2

    print(args.output_fname)
    # Build model and optimizer
    if str_to_boolean(args.do_train):
        train_data, valid_data = (
            make_data_for_discrimination(
                args.human_data_fname.format("train"),
                args.model_data_fname.format("train"),
                keymap[args.dataset],
                args.input_type,
            ),
            make_data_for_discrimination(
                args.human_data_fname.format("validation"),
                args.model_data_fname.format("validation"),
                keymap[args.dataset],
                args.input_type,
            ),
        )
        train_dataset = Dataset(args, tokenizer, train_data, args.input_type)
        dev_dataset = Dataset(args, tokenizer, valid_data, args.input_type)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size // args.accumulation,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=4,
        )
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=2,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=args.num_labels
        )
        model.to(device)
        model.train()
        t_total = len(train_loader) * args.epoch
        print("total training steps: ", t_total)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_total / 10),
            num_training_steps=t_total,
        )

        # Train model
        lowest_valid_loss = 9999.0
        global_step = 0
        for epoch in range(args.epoch):
            with tqdm(train_loader, unit="batch") as tepoch:
                for (
                    iteration,
                    (
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        position_ids,
                        labels,
                    ),
                ) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    position_ids = position_ids.to(device)
                    labels = labels.to(device, dtype=torch.long)

                    optimizer.zero_grad()

                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        labels=labels,
                    )

                    loss = output.loss
                    writer.add_scalar("loss/train", loss, global_step)
                    global_step += 1
                    loss.backward()

                    optimizer.step()
                    scheduler.step()

                    tepoch.set_postfix(loss=loss.item())

            # Evaluate the model five times per epoch
            with torch.no_grad():
                model.eval()
                valid_losses = []
                predictions = []
                target_labels = []
                for (
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    position_ids,
                    labels,
                ) in tqdm(
                    dev_loader,
                    desc="Eval",
                    position=1,
                    leave=None,
                ):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    position_ids = position_ids.to(device)
                    labels = labels.to(device, dtype=torch.long)

                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        labels=labels,
                    )

                    logits = output.logits
                    loss = output.loss
                    valid_losses.append(loss.item())
                    if args.num_labels != 2:
                        batch_predictions = [
                            int(torch.argmax(logit).cpu()) for logit in logits
                        ]
                    else:
                        batch_predictions = [
                            0 if example[0] > example[1] else 1 for example in logits
                        ]
                    batch_labels = [int(example) for example in labels]

                    predictions += batch_predictions
                    target_labels += batch_labels
            acc = compute_acc(predictions, target_labels)
            writer.add_scalar("acc/dev", acc, global_step)
            valid_loss = sum(valid_losses) / len(valid_losses)
            writer.add_scalar("loss/dev", valid_loss, global_step)
            if lowest_valid_loss > valid_loss:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.exp_dir, "model.pth"),
                )
                lowest_valid_loss = valid_loss
                print(
                    "Acc for model which have lower valid loss: ",
                    acc,
                )

    softmax_func = torch.nn.Softmax(dim=1)
    print("Do Test!")
    if str_to_boolean(args.do_test):
        assert not os.path.exists(args.output_fname)
        # Machine-generated texts에 대한 discrimination 진행 및 human-like 점수 저장
        model_test_data = make_data_for_scoring_by_discrimination(
            args.model_data_fname.format("train"), args.input_type
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=args.num_labels
        )
        model.load_state_dict(torch.load(os.path.join(args.exp_dir, "model.pth")))
        model.to(device)
        test_dataset = Dataset(args, tokenizer, model_test_data, args.input_type)
        print("The number of test samples:", len(test_dataset))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )
        model.eval()
        predictions = []
        with tqdm(test_loader, unit="batch") as test:
            for (
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                labels,
            ) in test:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                position_ids = position_ids.to(device)
                labels = labels.to(device, dtype=torch.long)
                with torch.no_grad():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                    )
                logits = output.logits
                if args.save_logits:
                    batch_predictions = [
                        [float(e) for e in logit] for logit in logits.cpu().numpy()
                    ]
                    predictions += batch_predictions

                else:
                    probs = softmax_func(logits)
                    batch_predictions = [
                        [float(e) for e in prob.cpu().numpy()][1] for prob in probs
                    ]
                    predictions += batch_predictions

        # Step2. Prediction and save
        assert len(test_dataset) == len(predictions)
        with open(args.output_fname, "w") as f:
            if args.save_logits:
                for e in predictions:
                    f.write("\t".join([str(score) for score in e]) + "\n")
            else:
                f.write("\n".join([str(e) for e in predictions]))


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
