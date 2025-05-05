FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3.8-dev python3.8-distutils \
    git wget curl build-essential cmake libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip setuptools==58.0.4

RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install \
    numba==0.48.0 \
    numpy==1.19.5 \
    open3d==0.14.1 \
    opencv_python==4.5.5.62 \
    PyYAML==6.0 \
    tensorboard \
    tqdm==4.62.3

WORKDIR /workspace

