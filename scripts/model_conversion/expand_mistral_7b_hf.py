import os
import sys
import math
import argparse
import sentencepiece as spm

from transformers import LlamaTokenizer, MistralConfig, MistralForCausalLM
from transformers.convert_slow_tokenizer import import_protobuf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="data/models/raw_hf/Mistral-7B-v0.1", help="Path to the checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="data/models/converted_hf/Mistral-7B-v0.1", help="Path to the output directory")
    args = parser.parse_args()

    # === Tokenizer ===
    new_tokens = [
        "<|im_start|>",
        "<|im_end|>",
    ]
    
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir)
    # NOTE: we don't do this as these new tokens will NOT be added to the underlying setence piece tokenizer!!!
    # tokenizer.add_tokens(new_tokens)
    # tokenizer.add_special_tokens({
    #     "additional_special_tokens": new_tokens
    # })
    tokenizer.save_pretrained(args.output_dir)

    # re-load the tokenizer to add the new tokens to the underlying sentence piece
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(args.output_dir)

    # a hack to modify the underlying sentence piece
    s = spm.SentencePieceProcessor()
    with open(tokenizer.vocab_file, "rb") as f:
        model_pb2 = import_protobuf()
        model = model_pb2.ModelProto.FromString(f.read())

        for token in new_tokens:
            new_token = model_pb2.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            model.pieces.append(new_token)

    with open(tokenizer.vocab_file, 'wb') as f:
        f.write(model.SerializeToString())

    # reload the tokenizer to reflect the changes
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(args.output_dir)
    
    # === Config ===
    config = MistralConfig.from_pretrained(args.ckpt_dir)
    print(config)

    # === Model ===
    model = MistralForCausalLM.from_pretrained(args.ckpt_dir, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # === Update config vocab size ===
    config.vocab_size = len(tokenizer) # len(tokenizer)
    print(config)

    # Save others to output_dir
    model.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
