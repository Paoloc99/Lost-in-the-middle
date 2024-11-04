#!/bin/bash

python evaluate_model.py \
--max_seq_length 1300 \
--dtype "None" \
--load_in_4bit True \
--use_liger True \
--model_name 'Paoloc99/litm_model_reg_1000_0' \
--output_dir 'results_metrics_1300' \
--filename 'kl_1000'