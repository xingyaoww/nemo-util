#!/bin/bash
echo "You supposed to run this script inside the docker container"

MODEL_PATH=/models/Mistral-7B-v0.1/
OUTPUT_NAME=mistral-7b-base.nemo
python3 NeMo/scripts/checkpoint_converters/convert_mistral_7b_hf_to_nemo.py \
    --input_name_or_path=$MODEL_PATH \
    --output_path=data/models/nemo/$OUTPUT_NAME \
    --precision=bf16
