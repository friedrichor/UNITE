PROMPT_TEMPLATE = {
    'RAW': {
        'text': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>",
        'image': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n<|endoftext|>",
        'video': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
    },
    'UNITE': {
        'base': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}\nSummary above {modalities} in one word:<|im_end|>\n<|im_start|>assistant\n<|endoftext|>",
        'text': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}\nSummary above sentence in one word:<|im_end|>\n<|im_start|>assistant\n<|endoftext|>",
        'image': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nSummary above image in one word:<|im_end|>\n<|im_start|>assistant\n<|endoftext|>",
        'video': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\nSummary above video in one word:<|im_end|>\n<|im_start|>assistant\n<|endoftext|>",
    }
}