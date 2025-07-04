import torch
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modeling_unite import UniteQwen2VL


model_path = 'friedrichor/Unite-Instruct-Qwen2-VL-7B'
model = UniteQwen2VL.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = UniteQwen2VL.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda",
#     attn_implementation='flash_attention_2', 
#     low_cpu_mem_usage=True,
# )

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)

def process_messages(messages):
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
        for msg in messages
    ]
    # print(texts)  # TODO: For easier understanding, you can output texts.
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts, 
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    return inputs

## ============================== Text-Image ==============================
messages_txt1 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "The book titled 'Riding with Reindeer - A Bicycle Odyssey through Finland, Lapland, and the Arctic' provides a detailed account of a journey that explores the regions of Lapland and the Arctic, focusing on the experience of riding with reindeer."},
            {"type": "text", "text": "\nSummary above sentence in one word:"},
        ],
    }
]

messages_txt2 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "The Cherished Teddies 116466 Best Teacher is a teddy bear holding a sign that proudly declares it as the best teacher."},
            {"type": "text", "text": "\nSummary above sentence in one word:"},
        ],
    }
]

messages_txt = [messages_txt1, messages_txt2]

messages_img1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./examples/518L0uDGe0L.jpg"},
            {"type": "text", "text": "\nSummary above image in one word:"},
        ],
    }
]

messages_img2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./examples/116466.jpg"},
            {"type": "text", "text": "\nSummary above image in one word:"},
        ],
    }
]

messages_img = [messages_img1, messages_img2]

inputs_txt = process_messages(messages_txt)
inputs_img = process_messages(messages_img)

with torch.no_grad():
    embeddings_txt = model(**inputs_txt)  # [2, 1536]
    embeddings_img = model(**inputs_img)  # [2, 1536]
    print(f"embeddings_txt.shape: {embeddings_txt.shape}")
    print(f"embeddings_img.shape: {embeddings_img.shape}")

    print(torch.matmul(embeddings_txt, embeddings_img.T))
    # tensor([[0.7578, 0.0270],
    #         [0.0459, 0.6406]], dtype=torch.bfloat16)
