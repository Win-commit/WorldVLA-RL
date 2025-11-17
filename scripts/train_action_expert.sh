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
STAGE="stage2"  
EXP_NAME="FeatureExpert_v1"
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


STAGE_ARGS="--stage stage2 --parallel_mode True --parallel_reward_groups 10 --reward_group_size 10 --gamma 0.9 --noise_factor 0.4"
FRAMES=1

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    train/train_action_expert.py \
    --dynamic_model_path /liujinxin/zhy/ICLR2026/logs/discard/after_VAE/STAGE1_BalanceLoss_StateNorm_ValueChunk_CVAE_EMA/checkpoint-8000 \
    ${STAGE_ARGS} \
    --action_expert_config /liujinxin/zhy/ICLR2026/models/action_patches/feature_expert_config.json \ 
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-5 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 30000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 5000 \
    --per_device_train_batch_size 6 \
    --frames ${FRAMES} \
    --action_frames 10 \
    --action_tokenizer_path "/liujinxin/zhy/ICLR2026/pretrain/fast" \
    --max_position_embeddings 6400 \
    --eval_strategy no \
    --seed 42 \
    --report_to "wandb" \
    --logging_steps 8 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 8 \
    --save_steps 1000 \
    --save_strategy "steps" \
    --remove_unused_columns False \
    --dataloader_pin_memory True \
    --dataloader_drop_last True \
    --exp_name $EXP_NAME 