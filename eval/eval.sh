#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python /liujinxin/zhy/ICLR2026/eval/libero_eval_v2.py --actor_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_EMABalance_StateNorm_CVAE_L1_Actor_8k_original_Videomode/checkpoint-10000" --env_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE1_BalanceLoss_StateNorm_ValueChunk_CVAE_EMA/checkpoint-8000" --parallel_reward_groups 10 --reward_group_size 10 --task_suite libero_10 --trials 50 --local_log_dir /liujinxin/zhy/ICLR2026/eval/logs/VAE &

# CUDA_VISIBLE_DEVICES=2 python /liujinxin/zhy/ICLR2026/eval/libero_eval_v2.py --actor_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_EMABalance_StateNorm_CVAE_L1_Actor_8k_original_Videomode/checkpoint-10000" --env_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE1_BalanceLoss_StateNorm_ValueChunk_CVAE_EMA/checkpoint-8000" --parallel_reward_groups 10 --reward_group_size 10 --task_suite libero_object --trials 50 --local_log_dir /liujinxin/zhy/ICLR2026/eval/logs/VAE &

# CUDA_VISIBLE_DEVICES=3 python /liujinxin/zhy/ICLR2026/eval/libero_eval_v2.py --actor_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE2_EMABalance_StateNorm_CVAE_L1_Actor_8k_original_Videomode/checkpoint-10000" --env_ckpt "/liujinxin/zhy/ICLR2026/logs/STAGE1_BalanceLoss_StateNorm_ValueChunk_CVAE_EMA/checkpoint-8000" --parallel_reward_groups 10 --reward_group_size 10 --task_suite libero_goal --trials 50 --local_log_dir /liujinxin/zhy/ICLR2026/eval/logs/VAE &
wait

