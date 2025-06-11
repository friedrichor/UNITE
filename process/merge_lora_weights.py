import os
import argparse

from unite.model import UniteQwen2VL
from unite.model.builder import load_pretrained_model


def main():
    parser = argparse.ArgumentParser(description='Merge and save model')
    parser.add_argument('--base_model', type=str, required=True, help='Path to base model')
    parser.add_argument('--adapter_model', type=str, required=True, help='Path to adapter model')
    args = parser.parse_args()

    base_model = args.base_model
    adapter_model = args.adapter_model

    if adapter_model.endswith('main'):
        merged_model = adapter_model.replace("main", "merged")
    else:
        merged_model = adapter_model + "-merged"

    tokenizer, model, processor = load_pretrained_model(
        model_path=adapter_model,
        model_base=base_model,
        attn_implementation=None, # attn_implementation="flash_attention_2",
    )

    print(model)

    tokenizer.save_pretrained(merged_model)
    processor.save_pretrained(merged_model)
    model.save_pretrained(merged_model)

if __name__ == "__main__":
    main()