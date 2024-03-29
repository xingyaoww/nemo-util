"""Convert OpenAI chat format to NeMo chat format.

Example usage:
python3 scripts/data/convert_openai_to_nemo_chat_format.py data/datasets/codeact.jsonl,data/datasets/general.jsonl --output_file data/datasets/codeact-mixture.nemo.jsonl

============
OpenAI chat format:
source = {
    'messages': [
        {'role': 'system', 'content': '{system message}'}, // this is optional
        {'role': 'user', 'content': '{turn 1 user message}'},
        {'role': 'assistant', 'content': '{turn 1 assistant message}'},
        {'role': 'user', 'content': '{turn 2 user message}'},
    ],
} 

Nemo chat format:
source = {
    'system': '{system message}',
    'conversations': [
        {'from': 'user', 'value': '{turn 1 user message}', 'label': None},
        {'from': 'assistant', 'value': '{turn 1 assistant message}', 'label': '{turn 1 assistant label}'},
        {'from': 'user', 'value': '{turn 2 user message}', 'label': None},
        {'from': 'assistant', 'value': '{turn 2 assistant message}', 'label': '{turn 2 assistant label}'},
    ],
    "mask": "user",
    "type": None,
}
"""

import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert OpenAI chat format to NeMo chat format')
parser.add_argument('input_file', type=str, help='Input file(s) in OpenAI chat format')
parser.add_argument('--output_file', type=str, default=None, help='Output file in NeMo chat format')
args = parser.parse_args()

def convert_openai_to_nemo_chat_format(openai_messages, mask_role='user'):
    nemo_instance = {}
    assert len(openai_messages) > 0, "OpenAI instance must have at least one conversation"
    if openai_messages[0]['role'] == 'system':
        nemo_instance['system'] = openai_messages[0]['content']
        openai_messages = openai_messages[1:]
    else:
        nemo_instance['system'] = None
    nemo_instance['conversations'] = []
    nemo_instance['mask'] = mask_role

    # https://github.com/xingyaoww/NeMo/blob/main/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py#L203-L206
    nemo_instance['type'] = None

    for message in openai_messages:
        if message['role'] == 'user':
            nemo_instance['conversations'].append({'from': 'user', 'value': message['content'], 'label': None})
        elif message['role'] == 'assistant':
            nemo_instance['conversations'].append({'from': 'assistant', 'value': message['content'], 'label': None})
        else:
            raise ValueError(f"Unknown role: {message['role']}")
    return nemo_instance

input_files = args.input_file.split(',')

n_total = 0
with open(args.output_file, 'w') as fout:
    for input_file in tqdm(input_files, desc='Converting files'):
        assert input_file.endswith('.jsonl'), "Input file must be in jsonl format"
        with open(input_file, 'r') as fin:
            for line in tqdm(fin, desc=f'Converting {input_file}'):
                openai_instance = json.loads(line)
                assert 'conversations' in openai_instance, "OpenAI instance must have 'conversations' key"
                nemo_instance = convert_openai_to_nemo_chat_format(openai_instance['conversations'])
                fout.write(json.dumps(nemo_instance) + '\n')
                n_total += 1
    print(f"Converted {input_file} to NeMo chat format and saved to {args.output_file}")

print(f"Converted {n_total} instances to NeMo chat format and saved to {args.output_file}")
