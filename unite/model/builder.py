import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoProcessor, BitsAndBytesConfig
import torch
from unite.model import *
from unite.utils import *


def load_pretrained_model(model_path, model_base, device_map="auto", attn_implementation="flash_attention_2", **kwargs):
    kwargs["device_map"] = device_map
    kwargs["torch_dtype"] = torch.bfloat16

    min_pixels = kwargs.pop('min_pixels', 256*28*28)
    max_pixels = kwargs.pop('max_pixels', 1280*28*28)

    processor_path = model_path
    # Load model
    if model_base is not None:
        if os.path.exists(os.path.join(model_path, "adapter_model.safetensors")):
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = UniteQwen2VL.from_pretrained(
                model_base, attn_implementation=attn_implementation, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.bfloat16)

            processor_path = model_base
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = UniteQwen2VL.from_pretrained(
                model_path, attn_implementation=attn_implementation, low_cpu_mem_usage=True, **kwargs
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = UniteQwen2VL.from_pretrained(
            model_path, attn_implementation=attn_implementation, low_cpu_mem_usage=True, **kwargs
        )

    processor = AutoProcessor.from_pretrained(
        processor_path,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    
    print_green(f"Model Class: {model.__class__.__name__}")

    return tokenizer, model, processor
