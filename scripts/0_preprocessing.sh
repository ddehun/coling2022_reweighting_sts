#! /bin/bash

# 1. Preparing sentences (C_src) for generating synthetic datasets by using DINO
for dataset in stsb qqp mrpc
do
    python preprocess_data.py --dataset $dataset
done

# Run below code after you locate PAWS dataset on "datasets/benchmarks/paws/dev_and_test.json" following https://github.com/google-research-datasets/paws .
python split_paws_dev_test.py
