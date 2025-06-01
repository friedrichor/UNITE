# Adopted from https://github.com/LLaVA-VL/LLaVA-NeXT. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from PIL import Image

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from unite.train.trainer import UniteTrainer
from unite.dataset import *
from unite.model import *
from unite.utils import *

import random
import numpy as np
import torch.backends.cudnn as cudnn

local_rank = None

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-VL-2B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: str = field(
        default=None, 
        metadata={"help": "Preprocessing mode", "choices": ["fused_base", "fused_w_targetmodal"]}
    )
    prompt_version: str = field(
        default=None, 
        metadata={"help": "prompt template version", "choices": ["RAW", "UNITE"]}
    )
    max_frames: int = field(default=8)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)  # TODO: IMPORTANT
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    force_save_steps: int = field(default=0, metadata={"help": "Step interval for forced checkpoint saving (0 to disable)"})
    force_save_dir: str = field(default="force_checkpoints", metadata={"help": "Directory for saving forced checkpoints"})
    temp: float = field(default=0.03)
    has_negative: bool = field(default=False)
    dual_loss: bool = field(default=True)
    target_modal_mask: bool = field(default=False)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['visual']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    rank0_print_green("safe_save_model_for_hf_trainer!!!")
    torch.cuda.synchronize()

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                processor: transformers.ProcessorMixin,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_print_green(f"data_args.lazy_preprocess: {data_args.lazy_preprocess}")

    if data_args.lazy_preprocess == 'fused_base':
        train_dataset_cls = FusedRetTrainDataset
        # data_collator_cls = FusedRetTrainDataCollator
        data_collator = FusedRetTrainDataCollator(tokenizer=tokenizer, processor=processor)
    elif data_args.lazy_preprocess == 'fused_w_targetmodal':
        train_dataset_cls = FusedTargetModalRetTrainDataset
        # data_collator_cls = FusedTargetModalRetTrainDataCollator
        data_collator = FusedTargetModalRetTrainDataCollator(tokenizer=tokenizer, processor=processor)
    else:
        raise ValueError(f"data_args.lazy_preprocess: {data_args.lazy_preprocess}")

    train_dataset = train_dataset_cls(
        data_path=data_args.data_path,
        processor=processor,
        prompt_version=data_args.prompt_version,
        max_frames=data_args.max_frames,
    )
    # data_collator = data_collator_cls(tokenizer=tokenizer, processor=processor)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    rank0_print_green(f"data_args: {data_args}")
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    rank0_print_green("world_size: ", training_args.world_size)
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    rank0_print_green("Load UniteQwen2VL...")
    model = UniteQwen2VL.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    rank0_print(f"init model: {model}")
    model.config.use_cache = False

    if training_args.gradient_checkpointing:  # TODO: required
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=256*28*28, max_pixels=1280*28*28
    )
    processor.tokenizer.padding_side = 'left'
    processor.tokenizer.truncation_side = 'left'  # TODO: change truncation_side to 'left', IMPORTANT
    processor.tokenizer.model_max_length = training_args.model_max_length
    tokenizer = processor.tokenizer
    
    rank0_print(f"tokenizer: {tokenizer}")
    rank0_print(f"model: {model}")

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              processor=processor,
                                              data_args=data_args)

    trainer = UniteTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        temp=training_args.temp, has_negative=training_args.has_negative,
        dual_loss=training_args.dual_loss, target_modal_mask=training_args.target_modal_mask,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
