#! /bin/bash

declare -a DataArray=("stsb" "mrpc" "qqp" 'paws') 
declare -a ModelArray=("robertabase-mean" "glove-avg" "robertabase-nli")

for task in "${DataArray[@]}"; do
    for model in "${ModelArray[@]}"; do
        python run_unsupervised_textual_similarity.py --task=$task --model_name=$model
    done
    python run_use.py --task=$task
done