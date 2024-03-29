#!/bin/bash

FILEPATH=data/datasets/codeact.nemo.jsonl
NEMO_MODEL=data/models/nemo/mistral-7b-base.nemo
MAX_SEQ_LEN=16384

OUTPUT_DIR=data/datasets_packed/codeact_mistral_7b_16k
mkdir -p $OUTPUT_DIR

# make all paths absolute
FILEPATH=$(realpath $FILEPATH)
NEMO_MODEL=$(realpath $NEMO_MODEL)
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

export PYTHONPATH=$(pwd)/NeMo:$(pwd)/Megatron-LM:$PYTHONPATH

pushd NeMo/
python scripts/nlp_language_modeling/prepare_packed_ft_dataset.py \
   model.data.train_ds.file_names=[$FILEPATH] \
   model.data.train_ds.max_seq_length=$MAX_SEQ_LEN \
   model.restore_from_path=$NEMO_MODEL \
   +output_dir=$OUTPUT_DIR \
   +pack_sizes=[$MAX_SEQ_LEN]
