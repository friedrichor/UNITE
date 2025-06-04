#!/bin/bash
set -x
# ======================================================================
## Configure MPI slots for multi-node training
# Set 8 slots per node to utilize all GPUs
sed -i "s/slots=1/slots=8/" /etc/mpi/hostfile
sed -i "s/slots=1/slots=8/" /etc/mpi/mpi-hostfile
hostfile="/etc/mpi/hostfile"
# ======================================================================
## Environment variables setup
export TOKENIZERS_PARALLELISM=false
export DISABLE_MLFLOW_INTEGRATION=true
export PYTHONPATH=`pwd`:$PYTHONPATH
# ======================================================================
### NCCL Configuration
# add your NCCL Configuration (if available)
# ======================================================================
ts=`date +%Y_%m_%d_%H_%M`
mkdir -p "./logs/train_adaptation"
# ======================================================================
nnodes=8
num_gpus=8
# ======================================================================
lora_r=8
lora_alpha=16
learning_rate=1e-4
temperature=0.03
num_train_epochs=1
per_device_train_batch_size=16
let GLOBAL_BATCH_SIZE=per_device_train_batch_size*num_gpus*nnodes

MAX_FRAMES=12
PROMPT_VERSION=UNITE
LAZY_PREPROCESS=fused_base
DUAL_LOSS=True
HAS_NEGATIVE=False
TARGET_MODAL_MASK=False
# ======================================================================
MODEL_PATH=Qwen/Qwen2-VL-7B-Instruct
DATA_PATH=./data/UNITE-Base-Data/unite_retrieval_adaptation.json

RUN_NAME=qwen2vl7b_stage1_lora_r${lora_r}_a${lora_alpha}_temp${temperature}_bs${GLOBAL_BATCH_SIZE}_lr${learning_rate}_ep${num_train_epochs}_pt${PROMPT_VERSION}_max${MAX_FRAMES}f
# ======================================================================
deepspeed --num_nodes ${nnodes} --num_gpus ${num_gpus} --hostfile ${hostfile} --master_port=10271 \
    unite/train/train_mem.py \
    --deepspeed ./scripts/ds_config/zero1.json \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --prompt_version $PROMPT_VERSION \
    --lazy_preprocess $LAZY_PREPROCESS \
    --temp $temperature \
    --max_frames $MAX_FRAMES \
    --has_negative $HAS_NEGATIVE \
    --dual_loss $DUAL_LOSS \
    --target_modal_mask $TARGET_MODAL_MASK \
    --lora_enable True \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout 0.05 \
    --bf16 True \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --output_dir ./checkpoints/${RUN_NAME}/main \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to none \
    --run_name ${RUN_NAME} 2>&1 | tee ./logs/train_adaptation/log_${ts}_${RUN_NAME}.log
wait
