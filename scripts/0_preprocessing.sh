#! /bin/bash

# 1. Preparing sentences (C_src) for generating synthetic datasets by using DINO
for dataset in stsb qqp mrpc
do
    python preprocess_data.py --dataset $dataset
done

