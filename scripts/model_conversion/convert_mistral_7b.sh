#!/bin/bash
echo "You supposed to run this script inside the docker container"

MODEL_PATH=/models/Mistral-7B-v0.1/
OUTPUT_NAME=mistral-7b-base.nemo

# Override the installed version of Megatron-LM, but use the cloned version
export PYTHONPATH=$(pwd)/NeMo:$(pwd)/Megatron-LM:$PYTHONPATH

python3 NeMo/scripts/checkpoint_converters/convert_mistral_7b_hf_to_nemo.py \
    --input_name_or_path=$MODEL_PATH \
    --output_path=data/models/nemo/$OUTPUT_NAME \
    --precision=bf16-mixed
