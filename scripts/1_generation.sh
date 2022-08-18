#! /bin/bash

###
# 1. Generating synthetic dataset using DINO and post-processing it.
# 2. Training a discriminator and scoring synthetic examples.
###

# sts
python3 dino.py --output_dir data --task_file task_specs/stsb-x2.json --input_file data/stsb-x1-dataset.train.jsonl --input_file_type jsonl --num_entries_per_input_and_label 5 --remove_duplicates --splitname=train
python3 dino.py --output_dir data --task_file task_specs/stsb-x2.json --input_file data/stsb-x1-dataset.validation.jsonl --input_file_type jsonl --num_entries_per_input_and_label 5 --remove_duplicates --splitname=validation 

python3 postprocess_dataset.py --input_file=./data/sts-dataset.train.jsonl --output_file=./data/sts-dataset.postprocessed.train.jsonl
python3 postprocess_dataset.py --input_file=./data/sts-dataset.validation.jsonl --output_file=./data/sts-dataset.postprocessed.validation.jsonl
python discrimination.py --input_type=single --dataset=stsb

#mrpc
python3 dino.py --output_dir data --task_file task_specs/mrpc-x2-regression.json --input_file data/mrpc-x1-dataset.train.jsonl --input_file_type jsonl --num_entries_per_input_and_label 5 --remove_duplicates --splitname train
python3 dino.py --output_dir data --task_file task_specs/mrpc-x2-regression.json --input_file data/mrpc-x1-dataset.validation.jsonl --input_file_type jsonl --num_entries_per_input_and_label 5 --remove_duplicates --splitname validation

python3 postprocess_dataset.py --input_file=./data/mrpc-regression-dataset.train.jsonl --output_file=./data/mrpc-regression-dataset.postprocessed.train.jsonl
python3 postprocess_dataset.py --input_file=./data/mrpc-regression-dataset.validation.jsonl --output_file=./data/mrpc-regression-dataset.postprocessed.validation.jsonl 
python discrimination.py --input_type=single --dataset=qqp --data_cls_to_reg

#qqp
python3 dino.py --output_dir data --task_file task_specs/qqp-x2-regression.json --input_file data/qqp-x1-dataset.train.jsonl --input_file_type jsonl --num_entries_per_input_and_label 5 --remove_duplicates --splitname train
python3 dino.py --output_dir data --task_file task_specs/qqp-x2-regression.json --input_file data/qqp-x1-dataset.validation.jsonl --input_file_type jsonl --num_entries_per_input_and_label 5 --remove_duplicates --splitname validation

python3 postprocess_dataset.py --input_file=./data/qqp-regression-dataset.train.jsonl --output_file=./data/qqp-regression-dataset.postprocessed.train.jsonl
python3 postprocess_dataset.py --input_file=./data/qqp-regression-dataset.validation.jsonl --output_file=./data/qqp-regression-dataset.postprocessed.validation.jsonl 
python discrimination.py --input_type=single --dataset=mrpc --data_cls_to_reg
