#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python /liujinxin/zhy/ICLR2026/eval/libero_eval_unified.py --ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance/checkpoint-7000" --parallel_reward_groups 10 --reward_group_size 5 --task_suite libero_object --trials 50 &
# CUDA_VISIBLE_DEVICES=1 python /liujinxin/zhy/ICLR2026/eval/libero_eval_unified.py --ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance/checkpoint-7000" --parallel_reward_groups 10 --reward_group_size 5 --task_suite  libero_spatial --trials 50 &
CUDA_VISIBLE_DEVICES=0 python /liujinxin/zhy/ICLR2026/eval/libero_eval_v2.py --actor_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_EMABalance_StateNorm_Actor_UNIVLA8k_original_Videomode/checkpoint-5500" --env_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE1_BalanceLoss_StateNorm_ValueChunk/checkpoint-1500" --parallel_reward_groups 10 --reward_group_size 10 --task_suite libero_spatial --trials 50 --local_log_dir /liujinxin/zhy/ICLR2026/eval/logs &

CUDA_VISIBLE_DEVICES=1 python /liujinxin/zhy/ICLR2026/eval/libero_eval_v2.py --actor_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_EMABalance_StateNorm_Actor_UNIVLA8k_original_Videomode/checkpoint-5500" --env_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE1_BalanceLoss_StateNorm_ValueChunk/checkpoint-1500" --parallel_reward_groups 10 --reward_group_size 10 --task_suite libero_goal --trials 50 --local_log_dir /liujinxin/zhy/ICLR2026/eval/logs &

# CUDA_VISIBLE_DEVICES=2 python /liujinxin/zhy/ICLR2026/eval/libero_eval_v2.py --actor_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance_StateNorm_Actor_UNIVLA8k/checkpoint-10000" --env_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200" --parallel_reward_groups 50 --reward_group_size 5 --task_suite libero_spatial --trials 50 --local_log_dir /liujinxin/zhy/ICLR2026/eval/ablation/50_5_libero &

# CUDA_VISIBLE_DEVICES=3 python /liujinxin/zhy/ICLR2026/eval/libero_eval_v2.py --actor_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance_StateNorm_Actor_UNIVLA8k/checkpoint-10000" --env_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200" --parallel_reward_groups 10 --reward_group_size 5 --task_suite libero_goal --trials 50 &
wait

