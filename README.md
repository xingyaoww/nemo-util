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


