#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python /liujinxin/zhy/ICLR2026/eval/libero_eval_unified.py --ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance/checkpoint-7000" --parallel_reward_groups 10 --reward_group_size 5 --task_suite libero_object --trials 50 &
# CUDA_VISIBLE_DEVICES=1 python /liujinxin/zhy/ICLR2026/eval/libero_eval_unified.py --ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance/checkpoint-7000" --parallel_reward_groups 10 --reward_group_size 5 --task_suite  libero_spatial --trials 50 &
CUDA_VISIBLE_DEVICES=1 python /liujinxin/zhy/ICLR2026/eval/libero_eval_v2.py --actor_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance_StateNorm_EnvActor/checkpoint-3000" --env_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200" --parallel_reward_groups 10 --reward_group_size 5 --task_suite libero_10 --trials 50 &
# CUDA_VISIBLE_DEVICES=0 python /liujinxin/zhy/ICLR2026/eval/libero_eval_unified.py --ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance/checkpoint-6500" --parallel_reward_groups 10 --reward_group_size 5 --task_suite  libero_object --trials 50 &
wait

