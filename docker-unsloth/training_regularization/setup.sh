#!/bin/bash

conda create --name train_env python=3.10 -y
conda activate train_env

pip install --upgrade pip setuptools wheel
pip install transformers==4.43.3
conda install -n train_env -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install matplotlib datasets
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install liger-kernel wandb