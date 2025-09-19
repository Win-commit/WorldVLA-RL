#!/bin/bash
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

export WANDB_BASE_URL="https://api.bandw.top"
API_KEY=bf924aa39303a0d8808787e3777696c3626d4850
wandb login $API_KEY

DATAPATH='/liujinxin/zhy/ICLR2026/datasets/libero/data/meta/libero_all_norm_patched.pkl'
STAGE=${1:-stage2}  
EXP_NAME="STAGE2_TRAINER_STAGE1EMABalance_StateNorm_EnvActor"
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV_MODEL_PATH=/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200
ACTOR_MODEL_PATH=/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance_StateNorm_warmup/checkpoint-300
# 设置STAGE特定参数
if [ "$STAGE" = "stage1" ]; then
    STAGE_ARGS="--stage stage1 --parallel_mode False"
    FRAMES=6
else
    STAGE_ARGS="--stage stage2 --parallel_mode True --parallel_reward_groups 10 --reward_group_size 5 --gamma 0.9 --noise_factor 0.4 --p 1"
    FRAMES=1
fi

SERVER_PORTS=()
SERVER_PIDS=()

for GPU_ID in {0..7}; do
    PORT=$((8000 + GPU_ID))
    SERVER_PORTS+=($PORT)
    
    echo "Starting environment model server #$((GPU_ID+1)) on GPU $GPU_ID, port $PORT"
    python /liujinxin/zhy/ICLR2026/train/env_model_server.py \
        --model_path $ENV_MODEL_PATH \
        --port $PORT \
        --parallel_reward_groups 10 \
        --reward_group_size 5 \
        --gamma 0.9 \
        --noise_factor 0.4 \
        --p 1.0 \
        --attn_implementation eager \
        --gpu_id $GPU_ID &
    
    SERVER_PID=$!
    SERVER_PIDS+=($SERVER_PID)
    echo "Environment model server #$((GPU_ID+1)) started, PID: $SERVER_PID"
done

SERVER_LIST=$(IFS=,; echo "localhost:${SERVER_PORTS[*]}")
echo "Server list: $SERVER_LIST"
sleep 60

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    train/train_v2.py \
    --actor_model_path ${ACTOR_MODEL_PATH} \
    ${STAGE_ARGS} \
    --env_servers $SERVER_LIST \
    --deepspeed configs/deepspeed/zero3_offload.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 8e-5 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 10000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 50 \
    --per_device_train_batch_size 1 \
    --frames ${FRAMES} \
    --action_frames 10 \
    --attn_type "flash_attention_2" \
    --action_tokenizer_path "/liujinxin/zhy/ICLR2026/pretrain/fast" \
    --max_position_embeddings 6400 \
    --eval_strategy no \
    --seed 42 \
    --logging_steps 8 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps  10 \
    --save_steps 500 \
    --save_strategy "steps" \
    --evaluation_strategy "no" \
    --run_name "unified_${STAGE}_training_$(date +%Y%m%d_%H%M%S)" \
    --remove_unused_columns False \
    --dataloader_pin_memory True \
    --dataloader_drop_last True  \
    --exp_name "STAGE2_TRAINER_STAGE1EMABalance_StateNorm_EnvActor" \
    --report_to "wandb" 

for PID in "${SERVER_PIDS[@]}"; do
    echo "Killing server with PID: $PID"
    kill -9 $PID
done