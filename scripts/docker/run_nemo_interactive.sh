#!/bin/bash
DOCKER_IMG=nvcr.io/nvidia/nemo:24.01.01.framework

WORK_DIR=`pwd`
MODEL_DIR=TODO # Your model directory

docker run \
    --gpus all \
    -e UID=$(id -u) \
    --shm-size=2g \
    --net=host \
    --ulimit memlock=-1 \
    --rm -it \
    -v $MODEL_DIR:/models \
    -v ${PWD}:/workspace \
    -w /workspace \
    $DOCKER_IMG \
    bash -c "useradd --shell /bin/bash -u $UID -o -c '' -m nemo && usermod -aG root nemo && cd /workspace && su nemo -c 'git config --global credential.helper store' && su nemo"
