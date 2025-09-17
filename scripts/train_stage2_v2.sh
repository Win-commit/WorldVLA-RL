#!/bin/bash
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=8

export WANDB_BASE_URL="https://api.bandw.top"
# API_KEY=bf924aa39303a0d8808787e3777696c3626d4850
# wandb login $API_KEY

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

python train/env_model_server.py \
    --model_path ${ENV_MODEL_PATH} \
    --parallel_reward_groups 10 \
    --reward_group_size 5 \
    --gamma 0.9 \
    --noise_factor 0.4 \
    --p 1 \
    --attn_implementation "eager" \
    --gpu_id 0 \
    --port 8000\  &
SERVER_PID1=$!
echo "Environment Model Server #1 Started, PID: $SERVER_PID1"


python train/env_model_server.py \
    --model_path ${ENV_MODEL_PATH} \
    --parallel_reward_groups 10 \
    --reward_group_size 5 \
    --gamma 0.9 \
    --noise_factor 0.4 \
    --p 1 \
    --attn_implementation "eager" \
    --gpu_id 1 \
    --port 8001\  &
SERVER_PID2=$!
echo "Environment Modeling Server #2 Started, PID: $SERVER_PID2"


python train/env_model_server.py \
    --model_path ${ENV_MODEL_PATH} \
    --parallel_reward_groups 10 \
    --reward_group_size 5 \
    --gamma 0.9 \
    --noise_factor 0.4 \
    --p 1 \
    --attn_implementation "eager" \
    --gpu_id 2 \
    --port 8002\  &
SERVER_PID3=$!
echo "Environment Modeling Server #3 Started, PID: $SERVER_PID3"

python train/env_model_server.py \
    --model_path ${ENV_MODEL_PATH} \
    --parallel_reward_groups 10 \
    --reward_group_size 5 \
    --gamma 0.9 \
    --noise_factor 0.4 \
    --p 1 \
    --attn_implementation "eager" \
    --gpu_id 3 \
    --port 8003\  &
SERVER_PID4=$!
echo "Environment Modeling Server #4 Started, PID: $SERVER_PID4"

python train/env_model_server.py \
    --model_path ${ENV_MODEL_PATH} \
    --parallel_reward_groups 10 \
    --reward_group_size 5 \
    --gamma 0.9 \
    --noise_factor 0.4 \
    --p 1 \
    --attn_implementation "eager" \
    --gpu_id 4 \
    --port 8004\  &
SERVER_PID5=$!
echo "Environment Modeling Server #4 Started, PID: $SERVER_PID5"

python train/env_model_server.py \
    --model_path ${ENV_MODEL_PATH} \
    --parallel_reward_groups 10 \
    --reward_group_size 5 \
    --gamma 0.9 \
    --noise_factor 0.4 \
    --p 1 \
    --attn_implementation "eager" \
    --gpu_id 5 \
    --port 8005\  &
SERVER_PID6=$!
echo "Environment Modeling Server #6 Started, PID: $SERVER_PID6"

python train/env_model_server.py \
    --model_path ${ENV_MODEL_PATH} \
    --parallel_reward_groups 10 \
    --reward_group_size 5 \
    --gamma 0.9 \
    --noise_factor 0.4 \
    --p 1 \
    --attn_implementation "eager" \
    --gpu_id 6 \
    --port 8006\  &
SERVER_PID7=$!
echo "Environment Modeling Server #7 Started, PID: $SERVER_PID7"

python train/env_model_server.py \
    --model_path ${ENV_MODEL_PATH} \
    --parallel_reward_groups 10 \
    --reward_group_size 5 \
    --gamma 0.9 \
    --noise_factor 0.4 \
    --p 1 \
    --attn_implementation "eager" \
    --gpu_id 7 \
    --port 8007\  &
SERVER_PID8=$!
echo "Environment Modeling Server #8 Started, PID: $SERVER_PID8"


sleep 60

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    train/train_v2.py \
    --actor_model_path ${ACTOR_MODEL_PATH} \
    ${STAGE_ARGS} \
    --env_servers "localhost:8000,localhost:8001,localhost:8002,localhost:8003,localhost:8004,localhost:8005,localhost:8006,localhost:8007" \
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
    --per_device_train_batch_size 2 \
    --frames ${FRAMES} \
    --action_frames 10 \
    --attn_type "flash_attention_2" \
    --action_tokenizer_path "/liujinxin/zhy/ICLR2026/pretrain/fast" \
    --max_position_embeddings 6400 \
    --eval_strategy no \
    --seed 42 \
    --logging_steps 8 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps  8 \
    --save_steps 500 \
    --save_strategy "steps" \
    --evaluation_strategy "no" \
    --run_name "unified_${STAGE}_training_$(date +%Y%m%d_%H%M%S)" \
    --remove_unused_columns False \
    --dataloader_pin_memory True \
    --dataloader_drop_last True  \
    --exp_name "STAGE2_TRAINER_STAGE1EMABalance_StateNorm_EnvActor" \
    --report_to "wandb" 

kill $SERVER_PID1
kill $SERVER_PID2
kill $SERVER_PID3
kill $SERVER_PID4
kill $SERVER_PID5
kill $SERVER_PID6
kill $SERVER_PID7
kill $SERVER_PID8