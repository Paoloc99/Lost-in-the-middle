#!/bin/bash

python training_regularization.py \
--max_seq_length 2048 \
--dtype "None" \
--load_in_4bit True \
--use_liger True \
--gradient_checkpointing True \
--use_reentrant True \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--num_train_epochs 3 \
--mu 10000 \
--train_dataset_size 70000 \
--model_name 'unsloth/Qwen2-0.5B-Instruct-bnb-4bit'
