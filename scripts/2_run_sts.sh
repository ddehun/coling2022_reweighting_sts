#! /bin/bash

for seed in 42 # Add more seed number for repetitive experiments
do
    ###################################
    # DINO
    ###################################
    for train in stsb
    do
        python run_training.py --input_train_file data/$train-dataset.postprocessed.train.jsonl --input_dev_file data/$train-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --seed=$seed
    for valid in paws qqp mrpc  stsb paws_wiki
        do
            python run_training.py --input_train_file data/$train-dataset.postprocessed.train.jsonl --input_dev_file data/$train-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only  --seed=$seed
        done
    done

    for train in qqp mrpc
    do
        python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train   --data_cls_to_reg --seed=$seed
    for valid in qqp mrpc paws stsb paws_wiki
        do
            python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only   --data_cls_to_reg  --seed=$seed
        done
    done

    #######################################
    # RISE (STSb, single, temperature 0.5)
    #######################################
    for weight in temp_0.5
    do
    for type in single
    do
    for train in stsb
        do
        python run_training.py --input_train_file data/$train-dataset.postprocessed.train.jsonl --input_dev_file data/$train-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --disc_input_type=$type --machine_reweight --reweight_strategy $weight  --seed=$seed
        for valid in paws qqp mrpc  stsb paws_wiki
        do
        python run_training.py --input_train_file data/$train-dataset.postprocessed.train.jsonl --input_dev_file data/$train-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only --disc_input_type=$type --seed=$seed  --machine_reweight --reweight_strategy $weight  
        done
    done
    done
    done
    
    #######################################
    # RISE (QQP, single, temperature 0.9)
    #######################################
    for weight in temp_0.9
    do
    for type in single
    do
    for train in qqp 
        do
        python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --disc_input_type=$type --machine_reweight  --data_cls_to_reg --reweight_strategy $weight  --seed=$seed
        for valid in paws_wiki paws qqp mrpc  stsb
        do
        python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only --disc_input_type=$type --machine_reweight  --data_cls_to_reg --reweight_strategy $weight  --seed=$seed
        done
    done
    done
    done

    #######################################
    # RISE (MRPC, single, temperature 0.7)
    #######################################
    for weight in temp_0.7
    do
    for type in single
    do
    for train in mrpc
        do
        python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --disc_input_type=$type --machine_reweight  --data_cls_to_reg --reweight_strategy $weight  --seed=$seed
        for valid in paws_wiki paws qqp mrpc  stsb
        do
        python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only --disc_input_type=$type --machine_reweight  --data_cls_to_reg --reweight_strategy $weight  --seed=$seed
        done
    done
    done
    done

    #######################################
    # Ablation Begin~
    #######################################
    #######################################
    # Ablation 1. Reweighting w. U(0,1)
    #######################################
    for weight_ablation in "uniform" 
        do
        
        #######################################
        # STSb
        #######################################
        
        for type in single
        do
        for train in stsb
            do
            python run_training.py --input_train_file data/$train-dataset.postprocessed.train.jsonl --input_dev_file data/$train-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --disc_input_type=$type --machine_reweight  --seed=$seed --weight_ablation_strategy=$weight_ablation
            for valid in paws qqp mrpc  stsb paws_wiki
            do
            python run_training.py --input_train_file data/$train-dataset.postprocessed.train.jsonl --input_dev_file data/$train-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only --disc_input_type=$type --weight_ablation_strategy $weight_ablation   --seed=$seed  --machine_reweight
            done
        done
        done
        
        
        #######################################
        # QQP
        #######################################
        for type in single
        do
        for train in qqp 
            do
            python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --disc_input_type=$type --machine_reweight  --data_cls_to_reg  --seed=$seed --weight_ablation_strategy=$weight_ablation
            for valid in paws_wiki paws qqp mrpc  stsb
            do
            python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only --disc_input_type=$type --machine_reweight  --data_cls_to_reg  --seed=$seed --weight_ablation_strategy=$weight_ablation
            done
        done
        done

        #######################################
        # MRPC
        #######################################
        for type in single
        do
        for train in mrpc 
            do
            python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --disc_input_type=$type --machine_reweight  --data_cls_to_reg  --seed=$seed --weight_ablation_strategy=$weight_ablation
            for valid in paws_wiki paws qqp mrpc  stsb
            do
            python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only --disc_input_type=$type --machine_reweight  --data_cls_to_reg  --seed=$seed --weight_ablation_strategy=$weight_ablation
            done
        done
        done
        done


    ################################
    # Ablation 2. Filtering
    ################################
    for type in single
    do
    for train in qqp mrpc 
        do
        python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --disc_input_type=$type --filtering_ratio=0.1 --data_cls_to_reg
        for valid in paws qqp mrpc  stsb
        do
        python run_training.py --input_train_file data/$train-regression-dataset.postprocessed.train.jsonl --input_dev_file data/$train-regression-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only --disc_input_type=$type --filtering_ratio=0.1  --data_cls_to_reg
        done
    done
    #
    for train in stsb
        do
        python run_training.py --input_train_file data/$train-dataset.postprocessed.train.jsonl --input_dev_file data/$train-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --disc_input_type=$type --filtering_ratio=0.1
        for valid in paws qqp mrpc  stsb
        do
        python run_training.py --input_train_file data/$train-dataset.postprocessed.train.jsonl --input_dev_file data/$train-dataset.postprocessed.validation.jsonl --output_dir logs-$seed/$train/dino-x1/ --task=$train --eval_task=$valid --eval_only --disc_input_type=$type --filtering_ratio=0.1
        done
    done
    done
    #######################################
    # Ablation End
    #######################################


done