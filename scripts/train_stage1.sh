#!/bin/bash
export WANDB_BASE_URL="https://api.bandw.top"
API_KEY=bf924aa39303a0d8808787e3777696c3626d4850
wandb login $API_KEY

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

DATAPATH='/liujinxin/zhy/ICLR2026/datasets/libero/data/meta/libero_all_norm_patched.pkl'
STAGE="stage1"  
EXP_NAME="STAGE1_BalanceLoss_StateNorm_ValueChunk_CVAE_EMA_MSE"
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


STAGE_ARGS="--stage stage1 --parallel_mode False"
FRAMES=3

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    train/train.py \
    --model_path /liujinxin/zhy/ICLR2026/ckpts/checkpoint-0 \
    ${STAGE_ARGS} \
    --deepspeed configs/deepspeed/zero3_offload.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 50.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 8000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 50 \
    --per_device_train_batch_size 6 \
    --frames ${FRAMES} \
    --action_frames 10 \
    --attn_type "flash_attention_2" \
    --action_tokenizer_path "/liujinxin/zhy/ICLR2026/pretrain/fast" \
    --max_position_embeddings 6400 \
    --eval_strategy no \
    --seed 42 \
    --report_to "wandb" \
    --logging_steps 4 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 6 \
    --save_steps 500 \
    --save_strategy "steps" \
    --evaluation_strategy "no" \
    --run_name "unified_${STAGE}_training_$(date +%Y%m%d_%H%M%S)" \
    --remove_unused_columns False \
    --dataloader_pin_memory True \
    --dataloader_drop_last True \
    --exp_name $EXP_NAME \
    --resume_from_checkpoint "/liujinxin/zhy/ICLR2026/logs/STAGE1_BalanceLoss_StateNorm_ValueChunk_CVAE_EMA_MSE/checkpoint-4500"
