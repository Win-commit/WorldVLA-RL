import sys
sys.path.append("/liujinxin/zhy/ICLR2026/models/Emu3")
from emu3.mllm.configuration_emu3 import Emu3RewardConfig,Emu3MoEConfig
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from models.univla_rl_unified import Emu3UnifiedRewardModel
from models.Emu3.emu3.mllm.modeling_emu3 import Emu3MoE
import torch
from models.univla_rl_unified import Emu3UnifiedRewardModel
# Save init checkpoint
#===============================================
# config = Emu3RewardConfig()
# config.save_pretrained("/liujinxin/zhy/ICLR2026/ckpts/checkpoint-0")
# model_path = "/liujinxin/zhy/UniVLA/ckpts/UniVLA/WORLD_MODEL_POSTTRAIN"
# tokenizer = Emu3Tokenizer.from_pretrained(model_path,model_max_length = 6400,trust_remote_code=True)
# tokenizer.save_pretrained("/liujinxin/zhy/ICLR2026/ckpts/checkpoint-0")

# model = Emu3UnifiedRewardModel(config,tokenizer)
# config1 = Emu3MoEConfig.from_pretrained(model_path)
# model.model = Emu3MoE.from_pretrained(model_path,config=config1,attn_implementation = 'eager',torch_dtype = torch.bfloat16)
# model.save_pretrained("/liujinxin/zhy/ICLR2026/ckpts/checkpoint-0")


# Load Test
#===============================================
config = Emu3RewardConfig.from_pretrained("/liujinxin/zhy/ICLR2026/ckpts/checkpoint-0")
tokenizer = Emu3Tokenizer.from_pretrained("/liujinxin/zhy/ICLR2026/ckpts/checkpoint-0",
    model_max_length = 6400,
    trust_remote_code=True)
model = Emu3UnifiedRewardModel.from_pretrained(
        "/liujinxin/zhy/ICLR2026/ckpts/checkpoint-0",
        config=config,
        tokenizer=tokenizer,
        trust_remote_code=True,
        parallel_mode=False,
        parallel_reward_groups=10,
        reward_group_size=5,
        gamma=0,
        noise_factor=0,
        detach_selected_reward_hs=True,
        attn_implementation= "eager",
        torch_dtype=torch.bfloat16 
    )

print(model)

print(tokenizer.encode(tokenizer.rtg_token),tokenizer.encode(tokenizer.rwd_token))