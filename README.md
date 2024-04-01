# Setup

```
git clone https://github.com/xingyaoww/nemo-util
git submodule update --init --recursive
```

# SFT Training pipleine (use Mistral-7B as an example)

## Step 1: Convert a huggingface model to NEMO format

You should `git clone` the model to `data/models/raw_hf` first following huggingface's instruction, for example to `data/models/raw_hf/Mistral-7B-v0.1`.

You should add chatML tokens to the model's tokenizer and re-size the model's embedding for subsequent SFT.

```bash
python3 scripts/model_conversion/expand_mistral_7b_tokenizer.py \
    --ckpt_dir data/models/raw_hf/Mistral-7B-v0.1 \
    --output_dir data/models/converted_hf/Mistral-7B-v0.1
```

You will see the converted model at `data/models/converted_hf/Mistral-7B-v0.1`. Now you can convert it to NEMO format:

```bash
# enter NEMO docker
MODEL_DIR=`pwd`/data/models/converted_hf ./scripts/docker/run_nemo_interactive.sh
# Do the conversion
./scripts/model_conversion/convert_mistral_7b.sh
```

Then you will be able to see your model at `data/models/nemo/mistral-7b-base.nemo`.

## Step 2: Prepare your dataset

We will use [CodeActInstruct](https://huggingface.co/datasets/xingyaoww/code-act) for example. You can first download it to `.jsonl` files:

```bash
python3 scripts/data/download_dataset_from_hf.py
```

Because CodeActInstruct uses OpenAI messages format for chat, you need to convert it to NeMo's chat format. Comparison between two format can be found [here](./scripts/data/convert_openai_to_nemo_chat_format.py).

```bash
python3 scripts/data/convert_openai_to_nemo_chat_format.py \
    data/datasets/codeact.jsonl,data/datasets/general.jsonl \
    --output_file data/datasets/codeact-mixture.nemo.jsonl
```

Then you can pack shorter examples in `data/datasets/codeact-mixture.nemo.jsonl` to a longer sequence by running:

```bash
./scripts/data/convert_nemo_chat_to_packed.sh
```

This script by default uses ChatML chat template and max sequence length of 16k. You can customize the script to better suite your need.

