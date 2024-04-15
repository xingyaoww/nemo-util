#!/bin/bash

NEMO_MODEL=data/models/nemo/mistral-7b-base.nemo

MAX_SEQ_LEN=16384
PACKED_DATA_FILEPATH=data/datasets_packed/codeact_mixture_mistral_7b_16k/packed_16384_seed42.npy
NUM_EXAMPLES=4409

EXP_ID=Mistral-7B-16k-sft-codeact
CONFIG_PATH=scripts/train/configs/$EXP_ID.yaml
OUTPUT_DIR=data/ckpts/$EXP_ID
mkdir -p $OUTPUT_DIR


GLOBAL_BATCH_SIZE=32
N_EPOCHS=5
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
STEPS_PER_EPOCH=$((NUM_EXAMPLES / GLOBAL_BATCH_SIZE))
echo "STEPS_PER_EPOCH: $STEPS_PER_EPOCH"
MAX_STEPS=$((NUM_EXAMPLES * N_EPOCHS / GLOBAL_BATCH_SIZE))
echo "MAX_STEPS: $MAX_STEPS"

# make all paths absolute
OUTPUT_DIR=$(realpath $OUTPUT_DIR)
NEMO_MODEL=$(realpath $NEMO_MODEL)
PACKED_DATA_FILEPATH=$(realpath $PACKED_DATA_FILEPATH)

CONFIG_PATH=$(realpath $CONFIG_PATH)
CONFIG_DIR=$(dirname $CONFIG_PATH)
CONFIG_NAME=$(basename $CONFIG_PATH .yaml)
echo "NEMO_MODEL: $NEMO_MODEL"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "CONFIG_DIR: $CONFIG_DIR"
export PYTHONPATH=$(pwd)/NeMo:$(pwd)/Megatron-LM:$PYTHONPATH

# export WANDB_DISABLED=True
export CUDA_DEVICE_MAX_CONNECTIONS=1

pushd NeMo/
# https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/sft.html#step-2-sft-training
python examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
   --config-path $CONFIG_DIR \
   --config-name $CONFIG_NAME \
   trainer.max_steps=$MAX_STEPS \
   exp_manager.exp_dir=$OUTPUT_DIR \
   exp_manager.checkpoint_callback_params.every_n_train_steps=$STEPS_PER_EPOCH \
   model.restore_from_path=$NEMO_MODEL \
   model.global_batch_size=$GLOBAL_BATCH_SIZE \
   model.data.train_ds.max_seq_length=$MAX_SEQ_LEN \
   model.data.train_ds.file_names=[$PACKED_DATA_FILEPATH] \
   'model.data.train_ds.concat_sampling_probabilities=[1.0]' \
