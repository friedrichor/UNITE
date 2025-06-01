import os
from os.path import basename
import json
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field

import random
import numpy as np
from itertools import chain
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import transformers
from unite.model.qwen_vl_utils import process_vision_info, fetch_image, fetch_video
from unite.prompt_template import PROMPT_TEMPLATE
from unite.constants import QWEN2VL_IMAGE_TOKENS, QWEN2VL_VIDEO_TOKENS, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from unite.utils import *


class ModalityEncoder:
    # Using bitwise operations to represent different modalities
    TEXT = 1    # 001
    IMAGE = 2   # 010 
    VIDEO = 4   # 100

    # Mapping from modality strings to bit values
    MODALITY_MAP = {
        "text": TEXT,
        "image": IMAGE,
        "video": VIDEO
    }

    @classmethod
    def encode(cls, modal_str: str) -> int:
        """Convert modality string to integer representation"""
        # Split string and remove whitespace from each modality string
        modalities = set(m.strip() for m in modal_str.split(','))
        encoded = 0
        for modality in modalities:
            if modality in cls.MODALITY_MAP:
                encoded |= cls.MODALITY_MAP[modality]
        return encoded

    @classmethod
    def decode(cls, encoded: int) -> str:
        """Convert integer representation back to modality string (for debugging)"""
        modalities = set()
        for modality, bit_value in cls.MODALITY_MAP.items():
            if encoded & bit_value:
                modalities.add(modality)
        return ",".join(sorted(modalities))  # Use "," as separator for consistent formatting


class FusedRetTrainDataset(Dataset):
    """
    [
        {
            "query": {
                "text": Optional[str], "{text}{vision_pad(optional)}"
                "image": Optional[Union[str, list[str]]], image_path
                "video": Optional(list[str]), video_path
            },
            "candidate": same format as query
            "negative": same format as query  (Optional)
            "target_modal": str  (Optional)
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
        try:
            data_dict = self.list_data_dict[idx]

            query_inputs = self.prepare_inputs_for_fusedmodal(data_dict['query'])
            candidate_inputs = self.prepare_inputs_for_fusedmodal(data_dict['candidate'])

            input_data = dict(
                query_inputs=query_inputs,
                candidate_inputs=candidate_inputs,
            )

            return input_data

        except Exception as e:
            print_red(f"[Error when FusedRetTrainDataset.__getitem__]\n{data_dict}: {str(e)}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


@dataclass
class FusedRetTrainDataCollator(object):

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.ProcessorMixin

    def __call__(self, instances: List[Dict[str, Dict[str, Any]]]):
        batch = dict()
        batch['query_inputs'] = self.processor(
            text=[ins['query_inputs']['text_inputs'] for ins in instances],
            images=list(chain.from_iterable(ins['query_inputs']['image_inputs'] for ins in instances if ins['query_inputs']['image_inputs'] is not None)) or None,
            videos=list(chain.from_iterable(ins['query_inputs']['video_inputs'] for ins in instances if ins['query_inputs']['video_inputs'] is not None)) or None,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=True, return_tensors="pt",
        )  # 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'
        batch['candidate_inputs'] = self.processor(
            text=[ins['candidate_inputs']['text_inputs'] for ins in instances],
            images=list(chain.from_iterable(ins['candidate_inputs']['image_inputs'] for ins in instances if ins['candidate_inputs']['image_inputs'] is not None)) or None,
            videos=list(chain.from_iterable(ins['candidate_inputs']['video_inputs'] for ins in instances if ins['candidate_inputs']['video_inputs'] is not None)) or None,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=True, return_tensors="pt",
        )  # 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'
        batch['target_modal'] = None
        batch['idx'] = None

        return batch


class FusedTargetModalRetTrainDataset(Dataset):
    def __init__(self, data_path, processor, prompt_version, max_frames, **kwargs):
        super().__init__(
            data_path=data_path,
            processor=processor,
            prompt_version=prompt_version,
            max_frames=max_frames,
            **kwargs
        )
    
    def __getitem__(self, idx):
        try:
            data_dict = self.list_data_dict[idx]

            query_inputs = self.prepare_inputs_for_fusedmodal(data_dict['query'])
            candidate_inputs = self.prepare_inputs_for_fusedmodal(data_dict['candidate'])
            target_modal = ModalityEncoder.encode(data_dict['target_modal'])

            input_data = dict(
                query_inputs=query_inputs,
                candidate_inputs=candidate_inputs,
                target_modal=target_modal,
            )

            return input_data

        except Exception as e:
            print_red(f"[Error when FusedTargetModalRetTrainDataset.__getitem__]\n{data_dict}: {str(e)}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


@dataclass
class FusedTargetModalRetTrainDataCollator(object):

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.ProcessorMixin

    def __call__(self, instances: List[Dict[str, Dict[str, Any]]]):
        batch = dict()
        batch['query_inputs'] = self.processor(
            text=[ins['query_inputs']['text_inputs'] for ins in instances],
            images=list(chain.from_iterable(ins['query_inputs']['image_inputs'] for ins in instances if ins['query_inputs']['image_inputs'] is not None)) or None,
            videos=list(chain.from_iterable(ins['query_inputs']['video_inputs'] for ins in instances if ins['query_inputs']['video_inputs'] is not None)) or None,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=True, return_tensors="pt",
        )  # 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'
        batch['candidate_inputs'] = self.processor(
            text=[ins['candidate_inputs']['text_inputs'] for ins in instances],
            images=list(chain.from_iterable(ins['candidate_inputs']['image_inputs'] for ins in instances if ins['candidate_inputs']['image_inputs'] is not None)) or None,
            videos=list(chain.from_iterable(ins['candidate_inputs']['video_inputs'] for ins in instances if ins['candidate_inputs']['video_inputs'] is not None)) or None,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=True, return_tensors="pt",
        )  # 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'
        batch['target_modal'] = torch.tensor([ins['target_modal'] for ins in instances])
        batch['idx'] = None

        return batch


## =============================================================================
class FusedHasNegRetTrainDataset(Dataset):
    def __init__(self, data_path, processor, prompt_version, max_frames, **kwargs):
        super().__init__(
            data_path=data_path,
            processor=processor,
            prompt_version=prompt_version,
            max_frames=max_frames,
            **kwargs
        )
    
    def __getitem__(self, idx):
        try:
            data_dict = self.list_data_dict[idx]

            query_inputs = self.prepare_inputs_for_fusedmodal(data_dict['query'])
            candidate_inputs = self.prepare_inputs_for_fusedmodal(data_dict['candidate'])
            negative_inputs = self.prepare_inputs_for_fusedmodal(data_dict['negative'])

            input_data = dict(
                query_inputs=query_inputs,
                candidate_inputs=candidate_inputs,
                negative_inputs=negative_inputs
            )

            return input_data

        except Exception as e:
            print_red(f"[Error when FusedHasNegRetTrainDataset.__getitem__]\n{data_dict}: {str(e)}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


@dataclass
class FusedHasNegRetTrainDataCollator(object):

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.ProcessorMixin

    def __call__(self, instances: List[Dict[str, Dict[str, Any]]]):
        batch = dict()
        batch['query_inputs'] = self.processor(
            text=[ins['query_inputs']['text_inputs'] for ins in instances],
            images=list(chain.from_iterable(ins['query_inputs']['image_inputs'] for ins in instances if ins['query_inputs']['image_inputs'] is not None)) or None,
            videos=list(chain.from_iterable(ins['query_inputs']['video_inputs'] for ins in instances if ins['query_inputs']['video_inputs'] is not None)) or None,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=True, return_tensors="pt",
        )  # 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'
        batch['candidate_inputs'] = self.processor(
            text=[ins['candidate_inputs']['text_inputs'] for ins in instances],
            images=list(chain.from_iterable(ins['candidate_inputs']['image_inputs'] for ins in instances if ins['candidate_inputs']['image_inputs'] is not None)) or None,
            videos=list(chain.from_iterable(ins['candidate_inputs']['video_inputs'] for ins in instances if ins['candidate_inputs']['video_inputs'] is not None)) or None,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=True, return_tensors="pt",
        )  # 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'
        batch['negative_inputs'] = self.processor(
            text=[ins['negative_inputs']['text_inputs'] for ins in instances],
            images=list(chain.from_iterable(ins['negative_inputs']['image_inputs'] for ins in instances if ins['negative_inputs']['image_inputs'] is not None)) or None,
            videos=list(chain.from_iterable(ins['negative_inputs']['video_inputs'] for ins in instances if ins['negative_inputs']['video_inputs'] is not None)) or None,
            truncation=True, max_length=self.tokenizer.model_max_length,
            padding=True, return_tensors="pt",
        )  # 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw'
        batch['target_modal'] = None
        batch['idx'] = None

        return batch
