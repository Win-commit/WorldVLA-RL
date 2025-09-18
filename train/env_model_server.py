# env_model_server.py
import torch
import socket
import json
import numpy as np
from models.univla_rl_unified import Emu3UnifiedRewardModel
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from models.Emu3.emu3.mllm.configuration_emu3 import Emu3RewardConfig
import websockets
import asyncio
import logging
import pickle  # 添加pickle模块用于二进制序列化

# 设置更详细的日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class EnvModelServer:
    def __init__(self, model_path, host="0.0.0.0", port=8765, 
                parallel_reward_groups=10, reward_group_size=5, 
                gamma=0.9, noise_factor=0.4, p=1.0,
                attn_implementation="eager", gpu_id=0):
        self.host = host
        self.port = port
        
        # 使用指定的GPU设备
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Server {port}] Using GPU device: {self.device}")
        
        # 加载环境模型
        logger.info(f"[Server {port}] Loading tokenizer from {model_path}")
        self.tokenizer = Emu3Tokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = Emu3RewardConfig.from_pretrained(model_path)
        
        logger.info(f"[Server {port}] Loading model from {model_path}")
        self.model = Emu3UnifiedRewardModel.from_pretrained(
            model_path,
            config=config,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            parallel_mode=True,
            parallel_reward_groups=parallel_reward_groups,
            reward_group_size=reward_group_size,
            gamma=gamma,
            noise_factor=noise_factor,
            p=p,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # 将模型移至指定GPU
        self.model.to(self.device)
        
        # 保存reward_group_size以便后续使用
        self.reward_group_size = reward_group_size
        
        logger.info(f"[Server {port}] Environment model loaded on {self.device}, "
                   f"config: parallel_reward_groups={parallel_reward_groups}, "
                   f"reward_group_size={reward_group_size}, gamma={gamma}, "
                   f"noise_factor={noise_factor}, p={p}")
        
        # 添加计数器跟踪请求
        self.request_count = 0
        self.successful_requests = 0
    
    # 修复方法签名，只接收一个参数
    async def handle_client(self, websocket):
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        request_id = self.request_count
        self.request_count += 1
        
        try:
            logger.info(f"[Server {self.port}] New connection from {client_id} (request #{request_id})")
            async for message in websocket:
                start_time = time.time()
                logger.info(f"[Server {self.port}] Processing request #{request_id} from {client_id}")
                
                # 接收二进制数据并反序列化
                data = pickle.loads(message)
                logger.info(f"[Server {self.port}] Request #{request_id}: Data received and deserialized, "
                           f"batch_size={len(data['text_ids_list'])}")
                
                text_ids_list = [torch.from_numpy(item) for item in data['text_ids_list']]
                image_token_ids = torch.from_numpy(data['image_token_ids'])
                states = torch.from_numpy(data['states'])
                
                # 移动数据到设备
                logger.info(f"[Server {self.port}] Request #{request_id}: Moving tensors to device {self.device}")
                text_ids_list = [x.to(self.device) for x in text_ids_list]
                image_token_ids = image_token_ids.to(self.device)
                states = states.to(self.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

                # 模型推理
                logger.info(f"[Server {self.port}] Request #{request_id}: Running model inference")
                with torch.no_grad():
                    reward_results = self.model.sample_rewards(
                        text_ids_list=text_ids_list,
                        image_token_ids=image_token_ids,
                        states=states
                    )
                
                # 优化: 仅发送关键hidden_states片段，而不是整个隐藏状态矩阵
                logger.info(f"[Server {self.port}] Request #{request_id}: Processing hidden states")
                hidden_states = reward_results['hidden_states'] 
                hidden_states_shape = hidden_states.shape 
                context_lengths = reward_results['context_lengths'] 
                best_reward_group = reward_results['best_reward_group'] 
                reward_group_size = self.reward_group_size 
                
                # 直接使用numpy数组而非列表，减少转换开销
                critical_segments = []
                for i in range(len(context_lengths)):
                    context_len_i = context_lengths[i]
                    best_idx = best_reward_group[i].item()
                    group_start = context_len_i + best_idx * (reward_group_size + 1)  # +1 for rtg
                    group_end = group_start + reward_group_size + 1  # +1 for rtg
                    critical_segment = hidden_states[i:i+1, group_start:group_end, :].cpu().to(torch.float32).numpy()
                    critical_segments.append(critical_segment)
                
                # 准备要发送的二进制数据
                logger.info(f"[Server {self.port}] Request #{request_id}: Preparing response data")
                serialized_results = {
                    'reward_preds_group_mean': reward_results['reward_preds_group_mean'].cpu().to(torch.float32).numpy(),
                    'best_reward_group': best_reward_group.cpu().numpy(),
                    'critical_segments': critical_segments,  # 只发送关键片段，已是numpy数组
                    'hidden_states_shape': hidden_states_shape,  # 完整形状信息，用于客户端重建
                    'context_lengths': context_lengths,
                    'reward_group_size': reward_group_size,
                    'noise_norm': float(reward_results['noise_norm']),
                    'reward_embedding_norm': float(reward_results['reward_embedding_norm']),
                    'rwd_noise_ratio': float(reward_results['rwd_noise_ratio']),
                    'rtg_noise_ratio': float(reward_results['rtg_noise_ratio'])
                }
                
                # 序列化为二进制数据
                binary_data = pickle.dumps(serialized_results)
                logger.info(f"[Server {self.port}] Request #{request_id}: Serialized response, size={len(binary_data)/(1024*1024):.2f}MB")
                
                # 发送二进制数据
                await websocket.send(binary_data)
                
                end_time = time.time()
                self.successful_requests += 1
                logger.info(f"[Server {self.port}] Request #{request_id} completed in {end_time-start_time:.2f}s "
                           f"({self.successful_requests}/{self.request_count} successful)")
                
        except Exception as e:
            logger.error(f"[Server {self.port}] Error handling client {client_id}, request #{request_id}: {e}")
            import traceback
            logger.error(f"[Server {self.port}] Traceback: {traceback.format_exc()}")
    
    async def start_server(self):
        # 增大最大消息大小并设置更大的超时时间
        logger.info(f"[Server {self.port}] Starting server on {self.host}:{self.port}")
        server = await websockets.serve(
            self.handle_client, 
            self.host, 
            self.port, 
            max_size=1024*1024*500,  # 500MB
            ping_interval=None,      # 禁用ping
            max_queue=100,           # 增大队列
            open_timeout=300,        # 5分钟握手超时
            close_timeout=300        # 5分钟关闭超时
        )
        logger.info(f"[Server {self.port}] Server started successfully, listening on {self.host}:{self.port}, "
                   f"max_size: 500MB, timeout: 300s, max_queue: 100")
        await server.wait_closed()

# 运行服务器
if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="环境模型服务器")
    parser.add_argument("--model_path", type=str, 
                        default="/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200",
                        help="环境模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机名")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口")
    
    # 添加模型相关参数
    parser.add_argument("--parallel_reward_groups", type=int, default=10, help="并行奖励组数")
    parser.add_argument("--reward_group_size", type=int, default=5, help="每组奖励token数量")
    parser.add_argument("--gamma", type=float, default=0.9, help="时间加权参数")
    parser.add_argument("--noise_factor", type=float, default=0.4, help="噪声因子")
    parser.add_argument("--p", type=float, default=1.0, help="采样概率")
    parser.add_argument("--attn_implementation", type=str, default="eager", help="注意力实现方式")
    parser.add_argument("--gpu_id", type=int, default=0, help="使用的GPU ID")
    
    args = parser.parse_args()
    
    logger.info(f"Starting environment model server on port {args.port} using GPU {args.gpu_id}")
    
    server = EnvModelServer(
        model_path=args.model_path, 
        host=args.host, 
        port=args.port,
        parallel_reward_groups=args.parallel_reward_groups,
        reward_group_size=args.reward_group_size,
        gamma=args.gamma,
        noise_factor=args.noise_factor,
        p=args.p,
        attn_implementation=args.attn_implementation,
        gpu_id=args.gpu_id
    )
    asyncio.run(server.start_server())