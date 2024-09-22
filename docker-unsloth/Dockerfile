# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set a fixed model cache directory.
ENV TORCH_HOME=/root/.cache/torch

# Install Python and necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential python3.10 python3-pip python3.10-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get install -y bash

# update pip and setuptools
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

# install PyTorch with CUDA 12.1 support and other essential packages
# use a dedicated conda env 
RUN conda create --name train_env python=3.10
RUN echo "source activate train_env" > ~/.bashrc
ENV PATH /opt/conda/envs/train_env/bin:$PATH

RUN pip install transformers==4.43.3
RUN conda install -n train_env -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
RUN pip install matplotlib datasets
RUN pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
RUN pip install liger-kernel wandb

# copy the fine-tuning script into the container

WORKDIR /

# endless running task to avoid container to be stopped
CMD [ "/bin/bash" , "-c", "tail -f /dev/null" ]
