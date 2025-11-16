import argparse
import os
import torch
import pickle
import numpy as np
from time import time
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
import sys
from PIL import Image
from torch.nn.functional import cross_entropy
import random
sys.path.append("/liujinxin/zhy/ICLR2026")
from models.univla_rl_unified import Emu3UnifiedRewardModel
from models.Emu3.emu3.mllm.processing_emu3 import Emu3Processor
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
import time

import transformers
from transformers.modeling_utils import PreTrainedModel


#========================================
# Huggingface 并没有完成对flex attention的支持，为了使用自己实现的flex attention，只能合理的越过检查
#===========================================
# 保存原始方法
original_autoset = PreTrainedModel._autoset_attn_implementation

# 创建补丁函数
def patched_autoset(cls, config, torch_dtype=None, device_map=None, check_device_map=True, use_flash_attention_2=False):
    # 如果是flex_attention，先改为eager通过检查，之后再改回来
    is_flex = config._attn_implementation == "flex_attention"
    if is_flex:
        config._attn_implementation = "eager"
    result = original_autoset(config, use_flash_attention_2, torch_dtype, device_map, check_device_map, )
    
    # 恢复flex_attention设置
    if is_flex:
        config._attn_implementation = "flex_attention"
    
    return result

# 应用补丁
PreTrainedModel._autoset_attn_implementation = classmethod(patched_autoset)



# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER/checkpoint-1500")
parser.add_argument('--data_path', type=str, default="/liujinxin/zhy/ICLR2026/datasets/libero/data/meta/libero_all_norm.pkl")
parser.add_argument('--vision_hub', type=str, default="/liujinxin/zhy/ICLR2026/pretrain/Emu3-VisionTokenizer")
parser.add_argument('--fast_path', type=str, default="/liujinxin/zhy/ICLR2026/pretrain/fast")
parser.add_argument('--action_predict_frame', type=int, default=10)
parser.add_argument('--use_gripper', type=bool, default=True)
parser.add_argument('--test_samples', type=int, default=2)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--parallel_mode', type=bool, default=True)
parser.add_argument('--parallel_reward_groups', type=int, default=5)
parser.add_argument('--reward_group_size', type=int, default=10)
parser.add_argument('--visual_token_pattern', type=str, default="<|visual token {token_id:0>6d}|>")
args = parser.parse_args()

# 加载数据
with open(args.data_path, 'rb') as f:
    train_meta = pickle.load(f)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
print("正在加载模型...")
tokenizer = Emu3Tokenizer.from_pretrained(
    args.model_path,
    padding_side="right",
    use_fast=False,
)

model = Emu3UnifiedRewardModel.from_pretrained(
    args.model_path,
    tokenizer=tokenizer,
    attn_implementation="flex_attention",
    torch_dtype=torch.bfloat16,
    parallel_mode=args.parallel_mode,
    parallel_reward_groups=args.parallel_reward_groups,
    reward_group_size=args.reward_group_size
).to(device)
model.eval()

# 加载视觉处理器和分词器
image_processor = AutoImageProcessor.from_pretrained(args.vision_hub, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(args.vision_hub, trust_remote_code=True).to(device).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
action_tokenizer = AutoProcessor.from_pretrained(args.fast_path, trust_remote_code=True)
image_processor.min_pixels = 80 * 80

# 辅助函数，与datasets.py中保持一致
def to_videostr(video_tokens: torch.Tensor, visual_token_pattern: str) -> str:
    frame_str_list = []
    for frame in video_tokens:
        frame_token_str = [
            visual_token_pattern.format(token_id=token_id)
            for token_id in frame.flatten()
        ]
        frame_str = "".join(frame_token_str)
        frame_str_list.append(frame_str)
    videostr = tokenizer.eof_token.join(frame_str_list)
    return videostr

def format_video_prompt(video_tokens: torch.Tensor, visual_token_pattern: str) -> str:
    frames, h, w = video_tokens.shape
    videostr = to_videostr(video_tokens, visual_token_pattern)
    video_prompt = (
        tokenizer.boi_token +
        f"{frames}*{h}*{w}" +
        tokenizer.img_token +
        videostr +
        tokenizer.eof_token +
        tokenizer.eoi_token
    )
    return video_prompt

# 设置评估参数
action_errors_per_dimension = [[] for _ in range(7)]

# 开始评估
print(f"开始评估 {args.test_samples} 个样本...")
for i in range(args.test_samples):
    # 随机选择任务
    # task_idx = random.randint(0, len(train_meta) - 1)
    task_idx = 50
    task_data = train_meta[task_idx]
    text = task_data['text']
    image_list = task_data['image']
    action_list = task_data['action']
    
    # 选择随机帧
    rand_idx = random.randint(0, len(image_list) - args.action_predict_frame - 1)
    image = image_list[rand_idx]
    
    # 处理视频
    video_code_raw = np.load(image)
    video_code_tensor = torch.from_numpy(video_code_raw)
    

    gripper_list = task_data['gripper_image']
    gripper = gripper_list[rand_idx]
    gripper_code_raw = np.load(gripper)
    gripper_code_tensor = torch.from_numpy(gripper_code_raw)
    
    # 正确处理图像token ids，与datasets.py一致
    main_prompt = format_video_prompt(video_code_tensor, args.visual_token_pattern)
    gripper_prompt = format_video_prompt(gripper_code_tensor, args.visual_token_pattern)
    merged_prompt = main_prompt + gripper_prompt
    image_token_ids = tokenizer(merged_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"].to(device)
    
    image_token_ids = image_token_ids.unsqueeze(0)
    
    # 获取真实动作
    action_gt = action_list[rand_idx:rand_idx + args.action_predict_frame]
    
    # 准备文本输入
    text_prompt = tokenizer.bos_token + text
    text_ids = tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"].to(device)
    text_ids_list = [text_ids[0]]
    
    # 正确处理states
    states = task_data['state'][rand_idx]
    states = torch.from_numpy(states).unsqueeze(0).unsqueeze(0).to(device)
    states = states.to(torch.bfloat16)
    
    # 使用模型进行推理
    with torch.no_grad():
        time_start = time.time()
        action_outputs = model.generate_actions_inference(
            text_ids_list=text_ids_list,
            image_token_ids=image_token_ids,
            states=states,
            action_tokenizer=action_tokenizer,
            max_new_tokens=50,
            action_vocab_size=action_tokenizer.vocab_size,
            action_dim=7,
            time_horizon=args.action_predict_frame,
            do_sample=False
        )
        time_end = time.time()
        print(f"推理时间: {time_end - time_start} 秒")
    # 获取预测动作
    action = action_outputs['actions']
    
    # 调试输出
    if args.debug:
        print(f"Task: {text}")
        print(f"Predicted Action Shape: {action.shape}")
        
        # 转换真实动作
        action_gt_id = action_tokenizer(action_gt)
        action_gt_decode = action_tokenizer.decode(action_gt_id, time_horizon=args.action_predict_frame, action_dim=7)
        print(f"Ground Truth Actions: {action_gt_decode[0]}")
        print(f"Predicted Actions: {action}")
        print(f"Best Reward Group: {action_outputs['best_reward_group'].item()}")
    
    # 计算每个维度上的误差
    min_len = min(action.shape[0], action_gt.shape[0])
    for t in range(min_len):
        for dim in range(action.shape[1]):
            error = np.abs(action[t, dim] - action_gt[t, dim])
            action_errors_per_dimension[dim].append(error.item())
    
    if (i + 1) % 50 == 0:
        print(f"已完成 {i+1}/{args.test_samples} 个样本的评估")

# 计算每个维度的平均误差
average_errors_per_dimension = [np.mean(errors) if errors else 0 for errors in action_errors_per_dimension]
print(f"每个动作维度的平均误差: {average_errors_per_dimension}")
print(f"总平均误差: {np.mean(average_errors_per_dimension)}") 