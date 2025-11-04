import os
# os.environ["MUJOCO_GL"] = "egl"
import torch
import pickle
import numpy as np
from time import time
import sys
from PIL import Image
from torch.nn.functional import cross_entropy
from random import shuffle
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
from collections import deque
import tqdm
import logging
import argparse
import imageio
import pdb
# 添加LIBERO路径
sys.path.append("/liujinxin/code/lhc/lerobot/")
sys.path.append("/liujinxin/code/lhc/lerobot/LIBERO/")
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import math
# 添加模型路径
sys.path.append("/liujinxin/zhy/ICLR2026")
from models.univla_rl_unified import Emu3UnifiedRewardModel
from models.Emu3.emu3.mllm.processing_emu3 import Emu3Processor
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from transformers import AutoModel, AutoImageProcessor, AutoProcessor

# 视频编码处理函数
def to_videostr(video_tokens: torch.Tensor, tokenizer, visual_token_pattern: str = "<|visual token {token_id:0>6d}|>") -> str:
    """将视频tokens转换为字符串格式"""
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

def format_video_prompt(video_tokens: torch.Tensor, tokenizer, visual_token_pattern: str = "<|visual token {token_id:0>6d}|>") -> str:
    """格式化视频提示符"""
    frames, h, w = video_tokens.shape
    videostr = to_videostr(video_tokens, tokenizer, visual_token_pattern)
    video_prompt = (
        tokenizer.boi_token +
        f"{frames}*{h}*{w}" +
        tokenizer.img_token +
        videostr +
        tokenizer.eof_token +
        tokenizer.eoi_token
    )
    return video_prompt

# 设置日志
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def log_message(message: str, log_file=None):
    """记录消息到控制台和日志文件"""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

# 任务套件枚举
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

# 每个任务套件的最大步数
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

# 每次预测的动作步数
NUM_ACTIONS_CHUNK = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class EvaluationConfig:
    # 模型相关参数
    pretrained_checkpoint: str = ""
    env_ckpt: Optional[str] = None
    vision_hub: str = "/liujinxin/zhy/ICLR2026/pretrain/Emu3-VisionTokenizer"
    fast_path: str = "/liujinxin/zhy/ICLR2026/pretrain/fast"
    parallel_mode: bool = True
    parallel_reward_groups: int = 10
    reward_group_size: int = 10
    num_open_loop_steps: int = 10
    visual_token_pattern: str = "<|visual token {token_id:0>6d}|>"
    noise_factor: float = 0.4
    gamma: float = 0.9
    
    # LIBERO环境参数
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    window_size: int = 1 #0就退化成了不加任何历史
    
    # 工具参数
    run_id_note: Optional[str] = None
    local_log_dir: str = "/liujinxin/zhy/ICLR2026/eval/logs"
    save_videos: bool = True
    use_wandb: bool = False
    wandb_entity: str = ""
    wandb_project: str = ""
    seed: int = 7
    

def setup_logging(cfg: EvaluationConfig):
    """设置日志记录"""
    # 创建运行ID
    ckpt_id = cfg.pretrained_checkpoint.split('-')[-1] if '-' in cfg.pretrained_checkpoint else "custom"
    run_id = f"UNIFIED-EVAL-{cfg.task_suite_name}-{DATE_TIME}-{ckpt_id}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    
    # 设置本地日志
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"记录日志到文件: {local_log_filepath}")
    
    # 初始化wandb日志记录（如果启用）
    if cfg.use_wandb:
        import wandb
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )
    
    return log_file, local_log_filepath, run_id

class HistoryManager:
    """管理图像和动作的历史记录"""
    def __init__(self, window_size=2):
        self.window_size = window_size
        self.vision_queue = deque(maxlen=self.window_size) # save gripper and main view image in a window
        self.state_queue = deque(maxlen=self.window_size) # save state in a window
        self.reward_queue = deque(maxlen=self.window_size) # save reward in a window
        self.action_queue = deque(maxlen=self.window_size) # save action in a window
    
    def add_image(self, image_inputs):
        """添加图像到历史队列"""
        self.vision_queue.append(image_inputs)
    
    def add_state(self, state):
        """添加状态到历史队列"""
        self.state_queue.append(state)
    
    def add_reward(self, reward):
        """添加奖励到历史队列"""
        self.reward_queue.append(reward)

    def get_history(self):
        """获取历史图像"""
        return {
            "vision": list(self.vision_queue),
            "state": list(self.state_queue),
            "reward": list(self.reward_queue),
            "action": list(self.action_queue)   
        }
    
    def add_action(self, action):
        """添加动作到历史队列"""
        self.action_queue.append(action)
    
    
    def reset(self):
        """重置历史队列"""
        self.vision_queue.clear()
        self.action_queue.clear()



def get_libero_env(task, resolution=256):
    """初始化并返回LIBERO环境"""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  
    return env, task_description


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, output_dir=None):
    """保存一个episode的MP4回放"""
    if output_dir is None:
        DATE = time.strftime("%Y_%m_%d")
        rollout_dir = f"./rollouts/{DATE}"
    else:
        rollout_dir = output_dir
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--unified--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    log_message(f"保存回放视频到路径 {mp4_path}", log_file)
    return mp4_path

def get_libero_dummy_action():
    """获取LIBERO环境的空动作"""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1]  # 7D动作空间

def get_libero_image(obs):
    """从观察中提取第三人称视角图像并预处理"""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # 重要：旋转180度以匹配训练预处理
    return img

def get_libero_wrist_image(obs):
    """从观察中提取手腕相机图像并预处理"""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # 重要：旋转180度以匹配训练预处理
    return img

def preprocess_libero_image(img_np, resize_size=(224, 224)):
    """预处理LIBERO图像"""
    # numpy -> PIL
    pil_img = Image.fromarray(img_np)
    # 调整大小
    pil_img = pil_img.resize(resize_size)
    # PIL -> numpy
    aug_img_np = np.array(pil_img)
    return aug_img_np


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def prepare_observation(obs, resize_size):
    """准备用于策略输入的观察"""

    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    
    img_aug = preprocess_libero_image(img, resize_size)
    wrist_img_aug = preprocess_libero_image(wrist_img, resize_size)

    robot_state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )
    #TODO: 归一化状态
    robot_state = normalize_state(robot_state)
    
    # 准备观察字典
    observation = {
        "full_image": img_aug,
        "wrist_image": wrist_img_aug,
        "robot_state": robot_state
    }
    
    return observation, img  

def normalize_state(state):
    """将状态归一化"""
    state_q01 = np.array([
        -0.3992278575897217,
        -0.26887813210487366,
        0.038001593202352524,
        1.508327841758728,
        -2.7210044860839844,
        -1.0806760787963867,
        0.0017351999413222075,
        -0.04002688080072403
        ])
    state_q99 = np.array([
        0.13545873761177063,
        0.3356631398200989,
        1.2704156637191772,
        3.277169704437256,
        2.4054365158081055,
        0.5977985858917236,
        0.04031208157539368,
        -0.001779157668352127
    ])
    normalized = 2 * (state - state_q01) / (state_q99 - state_q01 + 1e-8) - 1
    return np.clip(normalized, -1, 1)


def unormalize_action(action):
    """将归一化的动作转换回实际值"""
    action_high = np.array([
        0.93712500009996,
        0.86775000009256,
        0.93712500009996,
        0.13175314309916836,
        0.19275000005139997,
        0.3353504997073735,
        0.9996000000999599
    ])
    action_low = np.array([
        -0.7046250000751599,
        -0.80100000008544,
        -0.9375000001,
        -0.11467779149968735,
        -0.16395000004372,
        -0.2240490058320433,
        -1.0000000001
    ])
    action = 0.5 * (action + 1) * (action_high - action_low) + action_low
    return action

# def test_reward(reward_sampling_results,env_model):
#     h = reward_sampling_results['hidden_states']
#     context_len_i = reward_sampling_results['context_lengths'][0]
#     best_idx = reward_sampling_results['best_reward_group'][0].item()
#     group_start = context_len_i + best_idx * (5 + 1)
#     group_end = group_start + 5 + 1
#     selected_reward_hs = h[0:1, group_start:group_end, :]  # [1, G+1, H]
    
#     # 提取前5个向量 [5, H]
#     vectors = selected_reward_hs[0:1, :5, :]
#     rwd_hat = env_model.reward_head(vectors).squeeze(0)
#     rtg = env_model.rtg_head(selected_reward_hs[0:1, -1, :]).squeeze(0)
#     # 计算前5个向量的方差
#     variance = torch.var(rwd_hat, dim=0)  # 各维度的方差 [H]
#     total_variance = torch.sum(variance).item()  
    
#     # 还可以计算平均方差作为标准化指标
#     avg_variance = total_variance / rwd_hat.shape[1]
    
#     return {
#         "total_variance": total_variance,
#         "avg_variance": avg_variance,
#         "reward_hat": rwd_hat,
#         "rtg": rtg,
#     }

def image_level_enc_dec(images, image_tokenizer, image_processor, visual_token_pattern: str = "<|visual token {token_id:0>6d}|>"):
    """批量处理图像：编码并保存代码"""
    images_tensor = image_processor(images, return_tensors="pt")["pixel_values"].to(device)
    num_images = images_tensor.shape[0]
    for start_idx in range(0, num_images, 1): # Changed batch_size to 1 for simplicity
        batch = images_tensor[start_idx:start_idx + 1] # Process one image at a time
        try:
            with torch.no_grad():
                # 编码图像批次
                codes = image_tokenizer.encode(batch)
                return codes
        except Exception as e:
            print(f"处理起始于图像 {start_idx} 的批次时出错: {e}")

def get_action(observation, task_description, model, tokenizer, image_processor, image_tokenizer, processor, action_tokenizer, visual_token_pattern="<|visual token {token_id:0>6d}|>", env_model = None, log_file = None, history_manager = None):
    """使用模型获取动作"""
    # 编码图像
    video_code = image_level_enc_dec([observation["full_image"]], image_tokenizer=image_tokenizer, image_processor=image_processor, visual_token_pattern=visual_token_pattern)
    video_code_tensor = torch.from_numpy(video_code) if isinstance(video_code, np.ndarray) else video_code
    
    gripper_code = image_level_enc_dec([observation["wrist_image"]], image_tokenizer=image_tokenizer, image_processor=image_processor, visual_token_pattern=visual_token_pattern)
    gripper_code_tensor = torch.from_numpy(gripper_code) if isinstance(gripper_code, np.ndarray) else gripper_code
    
    # 正确格式化图像代码为模型所需的格式
    main_prompt = format_video_prompt(video_code_tensor, tokenizer, visual_token_pattern)
    gripper_prompt = format_video_prompt(gripper_code_tensor, tokenizer, visual_token_pattern)
    merged_prompt = main_prompt + gripper_prompt
    
    # 使用tokenizer将格式化的提示转换为token ids
    image_token_ids = tokenizer(merged_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"].to(device)
    image_token_ids = image_token_ids.unsqueeze(0)  # [1, 1, L_img]
        
    # 准备文本输入
    text_prompt = tokenizer.bos_token + task_description
    text_ids = tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")["input_ids"].to(device)
    text_ids_list = [text_ids[0]]
    
    # 准备状态输入
    robot_state = observation["robot_state"] #[8,]

    
    # 转换为tensor
    states = torch.tensor(robot_state, dtype=torch.bfloat16, device=device).unsqueeze(0).unsqueeze(0) #[1,1,8]
    
    # 使用模型进行推理
    with torch.no_grad():
        rewards = None
        if env_model is not None:
            rewards = env_model.sample_rewards(
                text_ids_list=text_ids_list,
                image_token_ids=image_token_ids,
                states=states,
                history = history_manager.get_history() if history_manager is not None else None
            )
        action_outputs = model.generate_actions_inference(
            text_ids_list=text_ids_list,
            image_token_ids=image_token_ids,
            states=states,
            reward_sampling_results=rewards,
            action_tokenizer=action_tokenizer,
            max_new_tokens=50,
            action_vocab_size=action_tokenizer.vocab_size,
            action_dim=7,
            time_horizon=NUM_ACTIONS_CHUNK,
            do_sample=False,
            history = history_manager.get_history() if history_manager is not None else None
        )
    
    if history_manager is not None:
        history_manager.add_image(image_token_ids)
        history_manager.add_state(states)
        history_manager.add_reward(rewards["critical_segments"] if rewards is not None else None)
        history_manager.add_action(action_outputs['action_ids'])
        
        
        
    # 处理动作输出
    actions = action_outputs['actions']
    
    # 转换为numpy数组
    actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
    
    # 对动作进行后处理
    actions_np = unormalize_action(actions_np)
    
    # 翻转抓取器动作
    actions_np[..., -1] = np.where(actions_np[..., -1] > 0, 1, -1)
    
    # 转换为列表
    actions_list = [actions_np[i].copy() for i in range(actions_np.shape[0])]
    
    return actions_list

def run_episode(
    cfg: EvaluationConfig,
    env,
    task_description: str,
    model,
    tokenizer,
    image_processor,
    image_tokenizer,
    processor,
    action_tokenizer,
    resize_size,
    initial_state=None,
    log_file=None,
    env_model = None,
):
    """在环境中运行单个episode"""
    # 重置环境
    env.reset()
    history_manager = HistoryManager(window_size=cfg.window_size)
    # 设置初始状态（如果提供）
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    
    # 初始化动作队列
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        log_message(f"警告: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) 与常量 NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) 不匹配！为获得最佳性能（速度和成功率），我们建议执行完整的动作块。", log_file)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    
    # 设置
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    
    # 运行episode
    success = False
    while t < max_steps + cfg.num_steps_wait:
        # 在前几个时间步什么都不做，让物体稳定下来
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action())
            t += 1
            continue
        
        # 准备观察
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)
        
        # 如果动作队列为空，重新查询模型
        if len(action_queue) == 0:
            # 查询模型获取动作
            actions = get_action(
                observation,
                task_description,
                model,
                tokenizer,
                image_processor,
                image_tokenizer,
                processor,
                action_tokenizer,
                cfg.visual_token_pattern,
                env_model,
                log_file,
                history_manager
            )
            action_queue.extend(actions)
        
        # 从队列获取动作
        action = action_queue.popleft()
        
        # 在环境中执行动作
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1
            
    
    return success, replay_images

def run_task(
    cfg: EvaluationConfig,
    task_suite,
    task_id: int,
    model,
    tokenizer,
    image_processor,
    image_tokenizer,
    processor,
    action_tokenizer,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    env_model = None,
):
    """运行单个任务的评估"""
    # 获取任务
    task = task_suite.get_task(task_id)
    
    # 获取初始状态
    initial_states = task_suite.get_task_init_states(task_id)
    
    # 初始化环境并获取任务描述
    env, task_description = get_libero_env(task, resolution=cfg.env_img_res)
    
    # 开始episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\n任务: {task_description}", log_file)
        
        # 处理初始状态
        if cfg.initial_states_path == "DEFAULT":
            # 使用默认初始状态
            initial_state = initial_states[episode_idx]
        log_message(f"开始episode {task_episodes + 1}...", log_file)
        
        # 运行episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            tokenizer,
            image_processor,
            image_tokenizer,
            processor,
            action_tokenizer,
            resize_size,
            initial_state,
            log_file,
            env_model
        )
        
        # 更新计数器
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1
        
        # 保存回放视频
        if cfg.save_videos:
            save_rollout_video(
                replay_images, total_episodes, success=success, 
                task_description=task_description, log_file=log_file,
                output_dir=os.path.join(cfg.local_log_dir, "videos", cfg.pretrained_checkpoint.split("/")[-1])
            )
        
        # 记录结果
        log_message(f"成功: {success}", log_file)
        log_message(f"目前完成的episodes总数: {total_episodes}", log_file)
        log_message(f"成功次数: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
    
    # 记录任务结果
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    
    log_message(f"当前任务成功率: {task_success_rate}", log_file)
    log_message(f"当前总成功率: {total_success_rate}", log_file)
    
    # 如果启用，记录到wandb
    if cfg.use_wandb:
        import wandb
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )
    
    return total_episodes, total_successes

def eval_libero(cfg: EvaluationConfig) -> float:
    """在LIBERO基准测试上评估模型"""
    # 加载模型和分词器
    log_message(f"加载模型从 {cfg.pretrained_checkpoint}...", None)
    
    tokenizer = Emu3Tokenizer.from_pretrained(
        cfg.pretrained_checkpoint,
        padding_side="right",
        use_fast=False,
    )
    
    model = Emu3UnifiedRewardModel.from_pretrained(
        cfg.pretrained_checkpoint,
        tokenizer=tokenizer,
        parallel_mode=cfg.parallel_mode,
        parallel_reward_groups=cfg.parallel_reward_groups,
        reward_group_size=cfg.reward_group_size,
        attn_implementation = "flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    
    env_model = None
    #============Env mode load=================
    if cfg.env_ckpt is not None:
        env_model = Emu3UnifiedRewardModel.from_pretrained(
            cfg.env_ckpt,
            tokenizer=tokenizer,
            parallel_mode=cfg.parallel_mode,
            parallel_reward_groups=cfg.parallel_reward_groups,
            reward_group_size=cfg.reward_group_size,
            attn_implementation = "eager",
            torch_dtype=torch.bfloat16,
            noise_factor = cfg.noise_factor,
            gamma = cfg.gamma
        ).to(device)
        print("env_model loaded")
        env_model.eval()
    
    # 加载视觉处理器和分词器
    image_processor = AutoImageProcessor.from_pretrained(cfg.vision_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(cfg.vision_hub, trust_remote_code=True).to(device).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    action_tokenizer = AutoProcessor.from_pretrained(cfg.fast_path, trust_remote_code=True)
    image_processor.min_pixels = 80 * 80
    
    # 设置图像大小
    resize_size = (256, 256)
    
    # 设置日志记录
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    log_message(f"任务套件: {cfg.task_suite_name}", log_file)
    
    # 初始化LIBERO任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    needToEvaluate = [0,1,2,3,4,5,6,7,8,9]
    # needToEvaluate = [8,9]
    # 开始评估
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        if task_id not in needToEvaluate:
            continue
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            tokenizer,
            image_processor,
            image_tokenizer,
            processor,
            action_tokenizer,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
            env_model
        )
    
    # 计算最终成功率
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    
    # 记录最终结果
    log_message("最终结果:", log_file)
    log_message(f"总episodes: {total_episodes}", log_file)
    log_message(f"总成功次数: {total_successes}", log_file)
    log_message(f"整体成功率: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    
    # 如果启用，记录到wandb
    if cfg.use_wandb:
        import wandb
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)
    
    # 关闭日志文件
    if log_file:
        log_file.close()
    
    return final_success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", type=str, default="libero_spatial", 
                      help="任务套件: libero_spatial, libero_object, libero_goal, libero_10")
    parser.add_argument("--actor_ckpt", type=str, default="/liujinxin/zhy/ICLR2026/logs/STAGE2_TRAINER_STAGE1EMABalance/checkpoint-3500",
                      help="模型检查点路径")
    parser.add_argument("--env_ckpt", type=str, default="")
    parser.add_argument("--parallel_mode", type=bool, default=True, 
                      help="是否使用并行奖励采样模式")
    parser.add_argument("--parallel_reward_groups", type=int, default=10,
                      help="并行奖励组数量")
    parser.add_argument("--reward_group_size", type=int, default=5,
                      help="每组奖励的token数")
    parser.add_argument("--trials", type=int, default=50,
                      help="每个任务的试验次数")
    parser.add_argument("--save_videos", type=bool, default=True,
                      help="是否保存回放视频")
    parser.add_argument("--visual_token_pattern", type=str, default="<|visual token {token_id:0>6d}|>",
                      help="视觉token的模式字符串")
    parser.add_argument("--local_log_dir", type=str, default="/liujinxin/zhy/ICLR2026/eval/logs",
                      help="本地日志目录")
    parser.add_argument("--noise_factor", type=float, default=0.4,
                      help="噪声因子")
    args = parser.parse_args()
    
    # 创建配置
    cfg = EvaluationConfig()
    cfg.pretrained_checkpoint = args.actor_ckpt
    if args.env_ckpt != "":
        cfg.env_ckpt = args.env_ckpt
    cfg.parallel_mode = args.parallel_mode
    cfg.parallel_reward_groups = args.parallel_reward_groups
    cfg.reward_group_size = args.reward_group_size
    cfg.num_trials_per_task = args.trials
    cfg.save_videos = True
    cfg.visual_token_pattern = args.visual_token_pattern
    cfg.local_log_dir = args.local_log_dir
    cfg.noise_factor = args.noise_factor
    # 设置任务套件
    if args.task_suite == "libero_spatial":
        cfg.task_suite_name = TaskSuite.LIBERO_SPATIAL
    elif args.task_suite == "libero_object":
        cfg.task_suite_name = TaskSuite.LIBERO_OBJECT
    elif args.task_suite == "libero_goal":
        cfg.task_suite_name = TaskSuite.LIBERO_GOAL
    else:
        cfg.task_suite_name = TaskSuite.LIBERO_10
    
    # 运行评估
    eval_libero(cfg) 