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

logging.basicConfig(level=logging.INFO)

class EnvModelServer:
    def __init__(self, model_path, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        
        # 加载环境模型
        self.tokenizer = Emu3Tokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = Emu3RewardConfig.from_pretrained(model_path)
        self.model = Emu3UnifiedRewardModel.from_pretrained(
            model_path,
            config=config,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            parallel_mode=True,
            parallel_reward_groups=10,
            reward_group_size=5,
            gamma=0.9,
            noise_factor=0.4,
            p=1.0,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16
        )
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 将模型移至GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logging.info(f"环境模型已加载，使用设备: {self.device}")
        
    async def handle_client(self, websocket, path):
        try:
            async for message in websocket:
                # 接收数据
                data = json.loads(message)
                
                # 处理输入数据
                text_ids_list = [torch.tensor(item) for item in data['text_ids_list']]
                image_token_ids = torch.tensor(data['image_token_ids'])
                states = torch.tensor(data['states'])
                
                # 移至正确设备
                text_ids_list = [x.to(self.device) for x in text_ids_list]
                image_token_ids = image_token_ids.to(self.device)
                states = states.to(self.device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                
                # 使用模型推理
                with torch.no_grad():
                    reward_results = self.model.sample_rewards(
                        text_ids_list=text_ids_list,
                        image_token_ids=image_token_ids,
                        states=states
                    )
                
                # 转换结果为可序列化格式
                serialized_results = {
                    'reward_preds_group_mean': reward_results['reward_preds_group_mean'].cpu().numpy().tolist(),
                    'best_reward_group': reward_results['best_reward_group'].cpu().numpy().tolist(),
                    'noise_norm': float(reward_results['noise_norm']),
                    'reward_embedding_norm': float(reward_results['reward_embedding_norm']),
                    'rwd_noise_ratio': float(reward_results['rwd_noise_ratio']),
                    'rtg_noise_ratio': float(reward_results['rtg_noise_ratio'])
                }
                
                # 发送结果
                await websocket.send(json.dumps(serialized_results))
                
        except Exception as e:
            logging.error(f"处理客户端请求时出错: {e}")
    
    async def start_server(self):
        async with websockets.serve(self.handle_client, self.host, self.port):
            logging.info(f"环境模型服务已启动，监听于 {self.host}:{self.port}")
            await asyncio.Future()  # 运行直到被取消

# 运行服务器
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="环境模型服务器")
    parser.add_argument("--model_path", type=str, 
                        default="/liujinxin/zhy/ICLR2026/logs/STAGE1_TRAINER_Balance_Loss_StateNorm/checkpoint-200",
                        help="环境模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机名")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口")
    
    args = parser.parse_args()
    
    server = EnvModelServer(args.model_path, args.host, args.port)
    asyncio.run(server.start_server())