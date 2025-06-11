export PYTHONPATH=`pwd`:$PYTHONPATH

BASE_MODEL=YOUR_BASE_MODEL_PATH  # e.g., Qwen/Qwen2-VL-2B-Instruct, friedrichor/Unite-Base-Qwen2-VL-2B
ADAPTER_MODEL=YOUR_LORA_PATH

python process/merge_lora_weights.py \
    --base_model $BASE_MODEL \
    --adapter_model $ADAPTER_MODEL
