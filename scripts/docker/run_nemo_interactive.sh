#!/bin/bash
# cd NeMo; docker build -t xingyaoww/nemo:main .
# docker push xingyaoww/nemo:main
DOCKER_IMG=xingyaoww/nemo:main
WORK_DIR=`pwd`

# read model directory from ENV VAR, if exists add it to the docker run command
if [ -z "$MODEL_DIR" ]; then
    echo "MODEL_DIR is not set. Please set MODEL_DIR to the directory containing the model files."
else
    echo "MODEL_DIR is set to '$MODEL_DIR'"
    EXTRA_ARGS="-v $MODEL_DIR:/models"
fi

docker run \
    --gpus all \
    -e UID=$(id -u) \
    --shm-size=2g \
    --net=host \
    --ulimit memlock=-1 \
    -e WANDB_API_KEY \
    --rm -it \
    $EXTRA_ARGS \
    -v ${PWD}:/workspace \
    -w /workspace \
    $DOCKER_IMG \
    bash -c "useradd --shell /bin/bash -u $UID -o -c '' -m nemo && usermod -aG root nemo && cd /workspace && su nemo -c 'git config --global credential.helper store' && su nemo"
