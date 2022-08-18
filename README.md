# Reweighting Strategy based on Synthetic Data Identification for Sentence Similarity

This repository is the code for Reweighting Strategy based on Synthetic Data Identification for Sentence Similarity (COLING2022).

## How to Begin

- Install required packages in ``requirements.txt``.
- Download benchmark datasets (STSb, QQP, and MRPC) from [this](TBA) drive link.
- Prepare PAWS-QQP dataset following [this](https://github.com/google-research-datasets/paws) repository, and locate it in ``datasets/benchmarks/paws/``.

## How to Reproduce

**1. Data preparation**

- Run ``scripts/0_preprocessing.sh`` script. This will prepare sentences (C_src) to make synthetic dataset, and split  PAWS dataset into dev and test splits.

**2. Synthetic dataset generation & Machine-written example identification**

- Run ``scripts/1_generation.sh`` script to generate synthetic examples and train a discriminator model that identifies them.
- A process to create synthetic datase is same with the original DINO fraework suggested by [Schick et al. (2021)](https://aclanthology.org/2021.emnlp-main.555/).

**3. Training and evaluating STS models**

- Run ``scripts/2_run_sts.sh`` to train bi-encoder models for sentence similarity tasks.
- The shell script is to reproduce all results in Table 2 (reweighting or not, ablation study).

**4. Other baseline models**

- Run ``scripts/3_run_other_baselines.sh`` to reprduce the results of other baseilne models in Table 6, such as GloVe, BERT, and USE.

## Acknolwedge

Codes to generate synthetic dataset are derieved from [Schick et al. (2021)](https://aclanthology.org/2021.emnlp-main.555/)'s work. ([Github](https://github.com/timoschick/dino))
