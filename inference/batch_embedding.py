import os
import json
import pickle
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Any

import numpy as np
from itertools import chain
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist

import transformers
from transformers import Trainer, TrainingArguments

from unite.model.qwen_vl_utils import process_vision_info, fetch_image, fetch_video
from unite.model.builder import load_pretrained_model
from unite.prompt_template import PROMPT_TEMPLATE
from unite.constants import QWEN2VL_IMAGE_TOKENS, QWEN2VL_VIDEO_TOKENS, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from unite.utils import *


local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    prompt_version: str = field(
        default=None, 
        metadata={"help": "prompt template version", "choices": ["RAW", "UNITE"]}
    )
    max_frames: int = field(default=8)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: Optional[bool] = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    deepspeed: Optional[str] = field(default=None)
    ddp_backend: str = field(default="nccl")
    output_path: str = field(default=None)


class FusedModalInferDataset(Dataset):
    """
    [
        {
            "text": Optional[str], "{text}{vision_pad(optional)}"
            "image": Optional[Union[str, list[str]]], image_path
            "video": Optional(list[str]), video_path
            "idx": str
        },
        ...
    ]
    """
    def __init__(
        self, 
        data_path: str,
        processor: transformers.ProcessorMixin,
        prompt_version: str,
        max_frames: int,
        **kwargs
    ):
        super().__init__()
        list_data_dict = json.load(open(data_path, "r"))
        
        self.list_data_dict = list_data_dict
        self.processor = processor
        self.prompt_template = PROMPT_TEMPLATE[prompt_version]

        self.image_placeholder = IMAGE_PLACEHOLDER
        self.video_placeholder = VIDEO_PLACEHOLDER

        self.max_frames = max_frames
    
    def process_image_content(self, image_input: str | list[str] | dict):
        if isinstance(image_input, str):  # image_path
            image_input = [image_input]
        elif isinstance(image_input, dict):  # {"type": "image", "image": image_path}
            image_input = [image_input['image']]
        return [
            {"type": "image", "image": image_path} for image_path in image_input
        ]
    
    def process_video_content(self, video_input: str | list[str] | dict):
        if isinstance(video_input, str):  # video_path
            video_input = [video_input]
        elif isinstance(video_input, dict):  # {"type": "video", "video": video_path}
            video_input = [video_input['video']]
        return [
            {"type": "video", "video": video_path, "max_pixels": 240 * 320, "fps": 1.0, "max_frames": self.max_frames, "min_frames": 4}
            for video_path in video_input
        ]
    
    def prepare_inputs_for_fusedmodal(self, input_info: dict[str, str | list[str]]):
        """
        {"text": Optional(str), "image": Optional(list[str]), "video": Optional(list[str])}
        """
        assert isinstance(input_info, dict), f"input_info is not Dict: {input_info}"
        modal_str_mapping = {'text': 'sentence', 'image': 'image', 'video': 'video'}
        modalities = [modal_str_mapping[key] for key in input_info if key in modal_str_mapping]

        has_text = ('text' in input_info)
        text = input_info['text'] if has_text else ""
        text = text.replace(self.image_placeholder, QWEN2VL_IMAGE_TOKENS).replace(self.video_placeholder, QWEN2VL_VIDEO_TOKENS)
        
        image_inputs, video_inputs = None, None
        if "image" in input_info:
            image_infos = self.process_image_content(input_info['image'])

            image_inputs = []
            for vision_info in image_infos:
                image_inputs.append(fetch_image(vision_info))
            assert image_inputs is not None, "Have [image], but image_inputs is None."

            if not has_text:
                text += QWEN2VL_IMAGE_TOKENS + "\n"
        
        if "video" in input_info:
            video_infos = self.process_video_content(input_info['video'])
            
            video_inputs = []
            for vision_info in video_infos:
                video_inputs.append(fetch_video(vision_info))
            assert video_inputs is not None, "Have [video], but video_inputs is None."

            if not has_text:
                text += QWEN2VL_VIDEO_TOKENS + "\n"
        
        text_inputs = self.prompt_template['base'].format(text=text.strip(), modalities=' and '.join(modalities))

        return dict(
            text_inputs=text_inputs,  # str
            image_inputs=image_inputs,  # None | list[torch.tensor]
            video_inputs=video_inputs,  # None | list[torch.tensor]
        )

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, idx):
        data_dict = self.list_data_dict[idx]

        inputs = self.prepare_inputs_for_fusedmodal(data_dict)
        input_data = {
            "inputs": inputs,
            "idx": data_dict["idx"]
        }

        return input_data


@dataclass
class FusedModalInferDataCollator(object):

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.ProcessorMixin

    def __call__(self, instances: List[Dict[str, Dict[str, Any]]]):
        batch = dict()
        batch['inputs'] = self.processor(
            text=[ins['inputs']['text_inputs'] for ins in instances],
            images=list(chain.from_iterable(ins['inputs']['image_inputs'] for ins in instances if ins['inputs']['image_inputs'] is not None)) or None,
            videos=list(chain.from_iterable(ins['inputs']['video_inputs'] for ins in instances if ins['inputs']['video_inputs'] is not None)) or None,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=True, return_tensors="pt",
        )  # 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'
        batch['idx'] = [ins['idx'] for ins in instances]

        return batch


class EmbeddingPredictor(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        model.eval()
        with torch.no_grad():
            embeddings = model(**inputs)
            
            # Process each pair of data individually and write to file
            for idx, embed in zip(inputs['idx'], embeddings):
                data_dict = {
                    'idx': idx,
                    'embedding': embed.cpu()
                }
                with open(self.args.output_path, 'ab') as f:
                    pickle.dump(data_dict, f)
            
            # Return None since we don't need to compute loss or other metrics
            return None, None, None


def predict():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    
    prompt_version = data_args.prompt_version
    prompt_template = PROMPT_TEMPLATE[prompt_version]

    os.makedirs(os.path.dirname(training_args.output_path), exist_ok=True)

    # load model and processor
    tokenizer, model, processor = load_pretrained_model(
        model_path=model_args.model_name_or_path,
        model_base=None,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
    )

    data_module = dict(
        eval_dataset=FusedModalInferDataset(
            processor=processor,
            output_path=training_args.output_path,
        ),
        data_collator=FusedModalInferDataCollator(tokenizer=tokenizer, processor=processor)
    )

    eval_dataset = data_module['eval_dataset']
    # Check the dataset
    rank0_print_green(f"Dataset size: {len(eval_dataset)}")
    rank0_print_green(f"First item: {eval_dataset[0]}")

    trainer = EmbeddingPredictor(
        model=model, args=training_args, tokenizer=tokenizer, **data_module
    )

    predictions = trainer.predict(eval_dataset)


if __name__ == "__main__":
    predict()


