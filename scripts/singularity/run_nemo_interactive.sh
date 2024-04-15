#!/bin/bash

WORK_DIR=`pwd`
IMAGE=nemo_main.sif
# only for verify correctness - since NCSA cuda driver version does not fully support it yet
echo "WORK_DIR=$WORK_DIR"
echo "IMAGE=$IMAGE"

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata
module load cuda/12.2.1

# read model directory from ENV VAR, if exists add it to the docker run command
if [ -z "$MODEL_DIR" ]; then
    echo "MODEL_DIR is not set. Please set MODEL_DIR to the directory containing the model files."
else
    echo "MODEL_DIR is set to '$MODEL_DIR'"
    EXTRA_ARGS="-v $MODEL_DIR:/models"
fi

apptainer run --nv \
    --no-home \
    --no-mount bind-paths \
    --cleanenv \
    --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --writable-tmpfs \
    --bind /scratch:/scratch \
    --bind $WORK_DIR:/workspace \
    --bind $MODEL_DIR:/models \
    $IMAGE \
    /bin/bash
