#!/bin/bash

PACKED_DATA_FILEPATH=data/datasets_packed/codeact_mixture_mistral_7b_32k/packed_32768_seed42.npy
NEMO_MODEL=data/models/nemo/mistral-7b-base.nemo
MAX_SEQ_LEN=32768

EXP_ID=Mistral-7B-16k-sft-codeact
CONFIG_PATH=scripts/train/configs/$EXP_ID.yaml
OUTPUT_DIR=data/ckpts/$EXP_ID
mkdir -p $OUTPUT_DIR


NUM_EXAMPLES=2290
GLOBAL_BATCH_SIZE=32
N_EPOCHS=5
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


export WANDB_DISABLED=True
export CUDA_DEVICE_MAX_CONNECTIONS=1

pushd NeMo/
# https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/sft.html#step-2-sft-training
python examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
   --config-path $CONFIG_DIR \
   --config-name $CONFIG_NAME \
   trainer.max_steps=$MAX_STEPS \
   exp_manager.exp_dir=$OUTPUT_DIR \
   model.restore_from_path=$NEMO_MODEL \
   model.global_batch_size=$GLOBAL_BATCH_SIZE \
   model.data.train_ds.file_names=[$PACKED_DATA_FILEPATH] \
   'model.data.train_ds.concat_sampling_probabilities=[1.0]' \

   

# # PEFT scheme: lora, ptuning, adapter, ia3, or none for full fineutning
# python examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
#    "model.micro_batch_size=1" \
#    "model.global_batch_size=128" \
#    +model.data.train_ds.packed_sequence=True \
#    "model.data.train_ds.file_names=[$PACKED_DATA_FILEPATH]" \
#    "model.data.validation_ds.file_names=null" \
#    +trainer.limit_val_batches=0 \
#    +trainer.limit_test_batches=0 \
#    model.restore_from_path=$NEMO_MODEL \
#    model.peft.peft_scheme='none' \
#    name=$EXP_ID \
#    exp_manager.exp_dir=$OUTPUT_DIR

#    # model.data.train_ds.file_names=[$FILEPATH] \
#    # model.data.train_ds.max_seq_length=$MAX_SEQ_LEN \
#    # model.restore_from_path=$NEMO_MODEL \
#    # +output_dir=$OUTPUT_DIR \
#    # +pack_sizes=[$MAX_SEQ_LEN] \
#    # +model.data.chat=True \
#    # '+model.data.chat_prompt_tokens.system_turn_start="<|im_start|>"' \
#    # '+model.data.chat_prompt_tokens.system_role="system"' \
#    # '+model.data.chat_prompt_tokens.turn_start="<|im_start|>"' \
#    # '+model.data.chat_prompt_tokens.label_start="<|im_start|>"' \
#    # '+model.data.chat_prompt_tokens.end_of_turn="<|im_end|>\n"' \
#    # '+model.data.chat_prompt_tokens.end_of_name="\n"' \
#    # '+model.data.chat_prompt_tokens.add_special_tokens=["<|im_start|>", "<|im_end|>"]'


   
#    # trainer.precision=bf16 \
#    # trainer.num_nodes=1 \
#    # trainer.devices=8 \
#    # trainer.sft.max_steps=-1 \
#    # trainer.sft.limit_val_batches=40 \
#    # trainer.sft.val_check_interval=1000 \
#    # model.megatron_amp_O2=True \
#    # model.restore_from_path=/path/to/your/mcore_gpt.nemo \
#    # model.optim.lr=5e-6 \
#    # model.answer_only_loss=True \
#    # model.data.num_workers=0 \
#    # model.data.train_ds.micro_batch_size=1 \
#    # model.data.train_ds.global_batch_size=128 \
#    # model.data.train_ds.file_path=/path/to/databricks-dolly-15k-output.jsonl \
#    # model.data.validation_ds.micro_batch_size=1 \
#    # model.data.validation_ds.global_batch_size=128 \
#    # model.data.validation_ds.file_path=/path/to/databricks-dolly-15k-output.jsonl \
#    # exp_manager.create_wandb_logger=True \
#    # exp_manager.explicit_log_dir=/results \
#    # exp_manager.wandb_logger_kwargs.project=sft_run \
#    # exp_manager.wandb_logger_kwargs.name=dolly_sft_run \
#    # exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
#    # exp_manager.resume_if_exists=True \
#    # exp_manager.resume_ignore_no_checkpoint=True \
#    # exp_manager.create_checkpoint_callback=True \
#    # exp_manager.checkpoint_callback_params.monitor=validation_loss
