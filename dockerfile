FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive # skip question

RUN apt-get update -y && \
    apt-get install -y wget libffi-dev gcc build-essential curl tcl-dev tk-dev uuid-dev lzma-dev liblzma-dev libssl-dev libsqlite3-dev -y && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz && \
    tar xzf Python-3.11.0.tgz && \
    cd Python-3.11.0 && \
    ./configure --enable-optimizations && \
    make install

WORKDIR /src

COPY pyproject.toml ./pyproject.toml

RUN python3 -m pip install -e .
RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118

COPY . ./code
RUN rm pyproject.toml
# RUN python3 -m pip install flash-attn --no-build-isolation
