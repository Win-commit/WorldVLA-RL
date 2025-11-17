import os
import random
from functools import partial
from copy import deepcopy
import torch
import time
from torch import nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, List
from torch.cuda.amp import autocast

# 添加分布式训练相关导入
import torch.distributed as dist

from models.reward_heads import ValueEncoder, ValueDecoder, reparameterize, vae_loss
from models.Emu3.emu3.mllm.modeling_emu3 import Emu3PreTrainedModel, Emu3Model
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from PIL import Image
from models.Projectors import ProprioProjector
from models.emu3_parallel_patch import apply_emu3_parallel_patch, generate_parallel_reward_attention_mask
from transformers import LogitsProcessor, GenerationConfig
from transformers.cache_utils import Cache
class Emu3UnifiedRewardModel(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, 
                 config, 
                 tokenizer: Emu3Tokenizer, 
                 parallel_mode: bool = False, 
                 parallel_reward_groups: int = 5, 
                 reward_group_size: int = 10,
                 gamma: float = 0.9,
                 noise_factor: float = 0.4
                 ):
        """
        Args:
            config: 模型配置
            tokenizer: Emu3分词器
            parallel_mode: 是否使用并行奖励采样模式（Stage2）
            parallel_reward_groups: 并行奖励组数（M组）
            reward_group_size: 每组奖励的token数（不包括rtg）
            p: Stage2执行奖励采样的概率
            gamma: 时间加权参数
            noise_factor: 噪声因子
        """
        super().__init__(config)
        self.tokenizer = tokenizer
        self.model: Emu3Model = Emu3Model(config)
        self.hidden_dim = config.hidden_size
        self.proprio = ProprioProjector(self.hidden_dim)
        
        # 模式配置
        self.parallel_mode = parallel_mode
        
        # 并行奖励采样参数
        self.parallel_reward_groups = parallel_reward_groups
        self.reward_group_size = reward_group_size


        # 噪声强度
        self.noise_factor = noise_factor

        # 时间加权参数
        self.gamma = gamma

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reward_encoder = ValueEncoder()
        self.rtg_encoder = ValueEncoder()
        self.reward_decoder = ValueDecoder()
        self.rtg_decoder = ValueDecoder()
        # 自动平衡Stage1损失
        self.auto_balance_stage1 = True
        self.loss_ema_decay = 0.99
        self.register_buffer('img_ce_ema', torch.tensor(1.0), persistent=False)
        self.register_buffer('rwd_mse_ema', torch.tensor(1.0), persistent=False)
        self.register_buffer('rtg_mse_ema', torch.tensor(1.0), persistent=False)
        
        # 初始化特殊token ID
        token_map = {
            'bos': tokenizer.bos_token,
            'state_beg': tokenizer.state_beg_token,
            'state_end': tokenizer.state_end_token,
            'rwd': tokenizer.rwd_token,
            'rwd_beg': tokenizer.rwd_beg_token if hasattr(tokenizer, 'rwd_beg_token') else None,
            'rwd_end': tokenizer.rwd_end_token if hasattr(tokenizer, 'rwd_end_token') else None,
            'rtg': tokenizer.rtg_token,
            'pad': tokenizer.pad_token,
            'boa': tokenizer.boa_token if hasattr(tokenizer, 'boa_token') else None,
            'eoa': tokenizer.eoa_token if hasattr(tokenizer, 'eoa_token') else None,
        }
        
        self.ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in token_map.items() if v is not None}
        
        # 如果是并行模式，应用并行奖励采样补丁
        if self.parallel_mode:
            apply_emu3_parallel_patch(self.model)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self,
        text_ids_list: List,
        image_token_ids: torch.LongTensor,
        states: torch.Tensor,
        reward_targets: Optional[torch.Tensor] = None,
        rtg_targets: Optional[torch.Tensor] = None,
        action_token_ids: Optional[List[torch.Tensor]] = None,
    ):
        if not self.parallel_mode:
            return self.forward_stage1(text_ids_list, image_token_ids, states, reward_targets, rtg_targets)
        else:
            raise NotImplementedError("Train stage1 not supported in parallel mode.")

    def forward_stage1(self,
        text_ids_list: List,
        image_token_ids: torch.LongTensor,
        states: torch.Tensor,
        reward_targets: Optional[torch.Tensor],
        rtg_targets: Optional[torch.Tensor],
    ):
        """
        Stage1前向传播（reward 感知）
        """
        B, K, L_img = image_token_ids.shape
        _, _, action_frames, _ = reward_targets.shape
        device = image_token_ids.device
        max_len = self.tokenizer.model_max_length
        
        # prepare embeddings
        static = {k: self.get_input_embeddings()(torch.full((1,1), v, device=device)) for k,v in self.ids.items() if k!='pad'}
        pad_emb = self.get_input_embeddings()(torch.full((1,1), self.ids['pad'], device=device))
        img_embs = self.get_input_embeddings()(image_token_ids)        # [B,K,L_img,H]
        state_embs = self.proprio(states)             # [B,K,H]

        seq_list = []
        mask_list = []
        label_list = []  # for CE over image tokens
        noise_norms = [] # record noise strength
        for i, text_ids in enumerate(text_ids_list):
            L_text = text_ids.size(0)
            # text embeddings and initial labels
            t_emb = self.get_input_embeddings()(text_ids.unsqueeze(0))  # [1,L_text,H]
            parts = [t_emb]
            labels = [-100] * L_text

            # per-frame segments
            for j in range(K):
                # image
                ids_ij = image_token_ids[i, j].tolist()
                emb_ij = img_embs[i:i+1, j]               # [1,L_img,H]
                parts.append(emb_ij)
                labels += ids_ij
                # state_beg, state, state_end
                parts += [static['state_beg'], state_embs[i:i+1,j:j+1,:], static['state_end']]
                labels += [-100, -100, -100]
                #reward
                for time_step in range(action_frames):
                    rwd_vector = static['rwd'].view(-1).float() 
                    q25 = torch.quantile(rwd_vector, 0.25)
                    q75 = torch.quantile(rwd_vector, 0.75)
                    noise = torch.rand(1, self.hidden_dim, device=device, dtype=static['rwd'].dtype) * (q75 - q25) + q25
                    noise = noise * self.noise_factor

                    noise_norms.append(torch.norm(noise).item())

                    parts += [static['rwd'] + noise]
                    parts += [static['rtg'] + noise]
                    labels += [-100, -100]


            # concat and pad to max_len
            seq_i = torch.cat(parts, dim=1)  # [1, L_i, H]
            labels_i = torch.tensor(labels, device=device, dtype=torch.long)
            
            L_i = seq_i.size(1)
            if L_i < max_len:
                pad_len = max_len - L_i
                seq_i = torch.cat([seq_i, pad_emb.expand(1,pad_len,self.hidden_dim)], dim=1)
                labels_i = torch.cat([labels_i, torch.full((pad_len,), -100, device=device)], dim=0)
                mask_i = torch.cat([torch.ones(L_i, device=device), torch.zeros(pad_len, device=device)], dim=0)
            else:
                seq_i = seq_i[:, :max_len, :]
                labels_i = labels_i[:max_len]
                mask_i = torch.ones(max_len, device=device)

            seq_list.append(seq_i)
            mask_list.append(mask_i.unsqueeze(0))      # [1, max_len]
            label_list.append(labels_i.unsqueeze(0))   # [1, max_len]

        # batch combine
        inputs_embeds = torch.cat(seq_list, dim=0)     # [B, max_len, H]
        attention_mask = torch.cat(mask_list, dim=0)   # [B, max_len]
        labels = torch.cat(label_list, dim=0)          # [B, max_len]

        # forward backbone
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        h = outputs.last_hidden_state                 # [B, max_len, H]
        logits = self.lm_head(h)

        # next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # compute losses
        img_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
        )
        # reward/rtg preds
        reward_positions = []
        rtg_positions = []
        # compute reward positions and preds
        for i, text_ids in enumerate(text_ids_list):
            # start just after  text
            pos = text_ids.size(0)
            for _ in range(K):
                # Skip image tokens
                pos += L_img
                # Skip state segment: state_beg + state token + state_end
                pos += 3
                # Now pos points to the reward token
                for time_step in range(action_frames):
                    reward_positions.append((i, pos))
                    rtg_positions.append((i, pos + 1))
                    pos += 2

        batch_indices = [p[0] for p in reward_positions]
        reward_pos_indices = [p[1] for p in reward_positions]
        rtg_pos_indices = [p[1] for p in rtg_positions]

        reward_vectors = h[batch_indices, reward_pos_indices] # [B*K*action_frames, hidden_dim]
        rtg_vectors = h[batch_indices, rtg_pos_indices] # [B*K*action_frames, hidden_dim]
        rwd_mse = 0
        rtg_mse = 0

        if reward_targets is not None and rtg_targets is not None:
            reward_targets_reshape = reward_targets.reshape(B*K*action_frames, -1)
            rtg_targets_reshape = rtg_targets.reshape(B*K*action_frames, -1)
            #Contional-VAE Encode:
            reward_mu, reward_log_var = self.reward_encoder(reward_vectors, reward_targets_reshape)
            rtg_mu, rtg_log_var = self.rtg_encoder(rtg_vectors, rtg_targets_reshape)
            z_reward = reparameterize(reward_mu, reward_log_var)
            z_rtg = reparameterize(rtg_mu, rtg_log_var)
            #Contional-VAE Decode:
            reward_preds = self.reward_decoder(reward_vectors, z_reward)
            rtg_preds = self.rtg_decoder(rtg_vectors, z_rtg)

            rwd_mse += vae_loss(reward_preds, reward_targets_reshape, reward_mu, reward_log_var)['total_loss']
            rtg_mse += vae_loss(rtg_preds, rtg_targets_reshape, rtg_mu, rtg_log_var)['total_loss']
        
        # 使用EMA对不同尺度的损失进行自适应加权
        if self.auto_balance_stage1:
            eps = 1e-8
            with torch.no_grad():
                self.img_ce_ema = self.img_ce_ema * self.loss_ema_decay + (1 - self.loss_ema_decay) * img_ce.detach()
                if reward_targets is not None:
                    self.rwd_mse_ema = self.rwd_mse_ema * self.loss_ema_decay + (1 - self.loss_ema_decay) * rwd_mse.detach()
                if rtg_targets is not None:
                    self.rtg_mse_ema = self.rtg_mse_ema * self.loss_ema_decay + (1 - self.loss_ema_decay) * rtg_mse.detach()
            total_loss = img_ce
            if reward_targets is not None:
                rwd_w = torch.clamp(self.img_ce_ema.detach() / (self.rwd_mse_ema.detach() + eps), 0.05, 10.0)
                total_loss = total_loss + rwd_w * rwd_mse
            if rtg_targets is not None:
                rtg_w = torch.clamp(self.img_ce_ema.detach() / (self.rtg_mse_ema.detach() + eps), 0.05, 10.0)
                total_loss = total_loss + rtg_w * rtg_mse
        else:
            total_loss = (0.005 * img_ce) + rwd_mse + rtg_mse
        
        avg_noise_norm = sum(noise_norms) / len(noise_norms)
        rwd_noise_ratio = avg_noise_norm / torch.norm(static['rwd']).item()
        rtg_noise_ratio = avg_noise_norm / torch.norm(static['rtg']).item()
        return {'loss': total_loss,
                'vision_loss': img_ce,
                'reward_loss': None if reward_targets is None else rwd_mse,
                'rtg_loss': None if rtg_targets is None else rtg_mse,
                'img_ce_ema': self.img_ce_ema,
                'rwd_mse_ema': self.rwd_mse_ema,
                'rtg_mse_ema': self.rtg_mse_ema,
                'balanced_rwd':torch.clamp(self.img_ce_ema.detach() / (self.rwd_mse_ema.detach() + 1e-8), 0.1, 10.0) * rwd_mse,
                'balanced_rtg':torch.clamp(self.img_ce_ema.detach() / (self.rtg_mse_ema.detach() + 1e-8), 0.1, 10.0) * rtg_mse,
                'noise_norm': avg_noise_norm,
                'rwd_noise_ratio': rwd_noise_ratio,
                'rtg_noise_ratio': rtg_noise_ratio,
                'logits': logits,
                'reward_preds': None,
                'rtg_pred': None,
                }

    def select_best_reward_groups(self, reward_preds, rtg_preds):
        """        
        Args:
            reward_preds: [B,K,M,10,14]
            rtg_preds: [B,K,M,10,14]
            
        Returns:
            best_group_indices: [B,K]
        """
        B, K, M, _, _ = reward_preds.shape
        device = reward_preds.device
        rewards = reward_preds[:, :, :, :self.reward_group_size, -1]  # [B,K,M,reward_group_size]
        final_rtg = rtg_preds[:, :, :, self.reward_group_size-1, -1]  # [B,K,M]
        
        time_weights = torch.pow(
            torch.tensor(self.gamma, device=device), 
            torch.arange(self.reward_group_size, device=device)
        )  # [reward_group_size]
        
        weighted_rewards = rewards * time_weights.view(1, 1, 1, -1)  # [B,K,M,reward_group_size]
        
        summed_rewards = weighted_rewards.sum(dim=-1)  # [B,K,M]
        rtg_weight = torch.pow(torch.tensor(self.gamma, device=device), torch.tensor(self.reward_group_size, device=device))
        final_scores = summed_rewards + rtg_weight * final_rtg  # [B,K,M]
        #=============Temp:临时修改，rtg已知为纯噪声，推理的时候先去掉他============
        final_scores = summed_rewards
        
        best_group_indices = torch.argmax(final_scores, dim=2)  # [B,K]
        
        return best_group_indices
    
    @torch.no_grad()
    def sample_rewards(self,
        text_ids_list: List,
        image_token_ids: torch.LongTensor,
        states: torch.Tensor,
        history: Optional[Dict] = None # for inference only
    ):
        """
        阶段2.1: 并行奖励采样部分，仅负责生成并行奖励组并选择最佳奖励组
        <text>+(<img>+<state_beg>+<state>+<state_end>+(<rwd_beg>+(<rwd>+<rtg>)*10+<rwd_end>)*M)*K
        Args:
            text_ids_list: 文本ID列表
            image_token_ids: 图像token IDs
            states: 机器人状态
            
        Returns:
            dict: 包含reward采样结果、最佳奖励组索引、隐藏状态等
            
        """
        B, K, L_img = image_token_ids.shape
        device = image_token_ids.device
        max_len = self.tokenizer.model_max_length
        
        # prepare embeddings
        static = {k: self.get_input_embeddings()(torch.full((1,1), v, device=device)) for k,v in self.ids.items() if k!='pad'}
        pad_emb = self.get_input_embeddings()(torch.full((1,1), self.ids['pad'], device=device))
        img_embs = self.get_input_embeddings()(image_token_ids)        # [B,K,L_img,H]
        state_embs = self.proprio(states)             # [B,K,H]
        
        # 计算reward嵌入的范数
        reward_scale = torch.norm(static['rwd']).item()
        rtg_scale = torch.norm(static['rtg']).item()
        
        # 记录所有样本的噪声强度
        all_noise_norms = []

        seq_list = []
        mask_list = []
        context_lengths = []  # Track context length for each sample
        last_image_token_pos = []  # Track last image token position for each sample
        for i, text_ids in enumerate(text_ids_list):
            context_length_ind = []
            L_text = text_ids.size(0)
            # text embeddings and initial labels
            t_emb = self.get_input_embeddings()(text_ids.unsqueeze(0))  # [1,L_text,H]
            parts = [t_emb]
            history_length = 0
            if history is not None:
                for history_time in range(len(history["vision"])):
                    parts += self.get_input_embeddings()(history["vision"][history_time])
                    history_length += history["vision"][history_time].shape[-1]
                    parts += [static['state_beg'], self.proprio(history["state"][history_time]), static['state_end']]
                    history_length += 1 + 1 + 1
                    parts += [static['rwd_beg']]
                    parts += [history["reward"][history_time][:,:,:].squeeze(1)]
                    parts += [static['rwd_end']]
                    history_length += 1 + history["reward"][history_time].squeeze(1).shape[1] + 1

            # per-frame segments (assuming K=1 for parallel reward sampling)
            for j in range(K):
                # image
                ids_ij = image_token_ids[i, j].tolist()
                emb_ij = img_embs[i:i+1, j]               # [1,L_img,H]
                parts.append(emb_ij)
                last_image_token_pos.append(len(parts)-1)
                # state_beg, state, state_end
                parts += [static['state_beg'], state_embs[i:i+1,j:j+1,:], static['state_end']]
                parts += [static['rwd_beg']]
                # Calculate context length (text + images + states)
                # Each group now has reward_group_size + 1 tokens (including rtg)
                context_len = L_text + (L_img + 3) + 1  # 3 for state_beg, state, state_end, 1 for rwd_beg
                if history is not None:
                    context_len += history_length 
                if context_length_ind == []:
                    context_length_ind.append(context_len)
                else:
                    group_len = self.parallel_reward_groups * 20 + 1
                    context_length_ind.append(context_length_ind[-1] + group_len + (L_img + 3) + 1)
            
                # Add parallel reward groups
                for group_idx in range(self.parallel_reward_groups):
                    rwd_vector = static['rwd'].view(-1).float() 
                    q25 = torch.quantile(rwd_vector, 0.25)
                    q75 = torch.quantile(rwd_vector, 0.75)
                    
                    # Sample uniformly between 25th and 75th percentiles
                    group_noise = torch.rand(1, self.hidden_dim, device=device, dtype=static['rwd'].dtype) * (q75 - q25) + q25
                    group_noise = group_noise * self.noise_factor
                    
                    # Calculate and record the current noise level
                    noise_norm = torch.norm(group_noise).item()
                    all_noise_norms.append(noise_norm)
                    
                    # Add reward tokens&rtg tokens with noise
                    for token_idx in range(10):
                        rwd_emb = static['rwd'] + group_noise
                        rtg_emb = static['rtg'] + group_noise
                        parts += [rwd_emb]
                        parts += [rtg_emb]
                    
                parts += [static['rwd_end']]

            # concat and pad to max_len
            seq_i = torch.cat(parts, dim=1)  # [1, L_i, H]
            context_lengths.append(context_length_ind)
            
            L_i = seq_i.size(1)
            if L_i < max_len:
                pad_len = max_len - L_i
                seq_i = torch.cat([seq_i, pad_emb.expand(1,pad_len,self.hidden_dim)], dim=1)
                mask_i = torch.cat([torch.ones(L_i, device=device), torch.zeros(pad_len, device=device)], dim=0)
            else:
                seq_i = seq_i[:, :max_len, :]
                mask_i = torch.ones(max_len, device=device)

            seq_list.append(seq_i)
            mask_list.append(mask_i.unsqueeze(0))      # [1, max_len]

        # batch combine
        inputs_embeds = torch.cat(seq_list, dim=0)     # [B, max_len, H]
        attention_mask = torch.cat(mask_list, dim=0)   # [B, max_len]

        # Generate custom 4D attention mask for parallel reward sampling
        seq_len = inputs_embeds.size(1)
        
        # Generate individual masks for each sample since context_len varies
        custom_4d_masks = []
        for i in range(B):
            context_length_ind = context_lengths[i]
            reverse_context_length_ind = context_length_ind[::-1]
            sample_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
            for prefix_length in reverse_context_length_ind:
                sample_mask_part = generate_parallel_reward_attention_mask(
                    seq_length=prefix_length + self.parallel_reward_groups * 20 + 1,
                    context_len=prefix_length,
                    reward_groups=self.parallel_reward_groups,
                    reward_group_size=20,
                    device=device
                )
                part_len = sample_mask_part.size(-1)
                sample_mask[:, :, :part_len, :part_len] = sample_mask_part

            # Apply padding mask for this sample
            valid_length = int(attention_mask[i].sum().item())
            if valid_length < seq_len:
                sample_mask[0, 0, :, valid_length:] = -1e9
                sample_mask[0, 0, valid_length:, :] = -1e9
            
            custom_4d_masks.append(sample_mask)
        
        # Stack all sample masks
        custom_4d_mask = torch.cat(custom_4d_masks, dim=0)  # [B, 1, seq_len, seq_len]

        # Apply custom 4D attention mask for parallel reward sampling
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            custom_4d_attention_mask=custom_4d_mask,
            return_dict=True,
            output_hidden_states=True
        )
        h = outputs.last_hidden_state                 # [B, max_len, H]
        logits = self.lm_head(h)

        # print(f"output_hidden_states len: {len(outputs.hidden_states)}") 33

        # reward/rtg preds with parallel sampling
        reward_preds = []
        reward_pos = []
        rtg_pos = []
        for i, text_ids in enumerate(text_ids_list):
            # Parallel reward sampling: predict from all reward groups
            for k in range(K):
                for group_idx in range(self.parallel_reward_groups):
                    group_start_pos = context_lengths[i][k] + group_idx * (20)   
                    
                    # Decode reward tokens
                    for token_idx in range(20):
                        current_pos = group_start_pos + token_idx
                        if token_idx % 2 == 0:
                            reward_pos.append((i, current_pos))
                        else:
                            rtg_pos.append((i, current_pos))
        
        batch_indices = [p[0] for p in reward_pos]
        reward_pos_indices = [p[1] for p in reward_pos] # [B*K*M*10]
        rtg_pos_indices = [p[1] for p in rtg_pos]

        reward_vectors = h[batch_indices, reward_pos_indices] # [B*K*M*10, hidden_dim]
        rtg_vectors = h[batch_indices, rtg_pos_indices] # [B*K*M*10, hidden_dim]
        
        #Value decode
        z_latent = torch.randn(B*K*self.parallel_reward_groups*10, 14, dtype=reward_vectors.dtype, device=device)
        reward_preds = self.reward_decoder(reward_vectors, z_latent).view(B, K, self.parallel_reward_groups, 10, -1) # [B,K,M,10,14]
        rtg_preds = self.rtg_decoder(rtg_vectors, z_latent).view(B, K, self.parallel_reward_groups, 10, -1) # [B,K,M,10,14]

        best_group_indices = self.select_best_reward_groups(reward_preds, rtg_preds)
            
        h = h.detach()
            
        # 计算平均噪声强度
        avg_noise_norm = sum(all_noise_norms) / len(all_noise_norms) if all_noise_norms else 0
        rwd_noise_ratio = avg_noise_norm / reward_scale if reward_scale > 0 else 0
        rtg_noise_ratio = avg_noise_norm / rtg_scale if rtg_scale > 0 else 0
        
        #方便后续使用，直接把需要的reward和rtg返回
        critical_segments = []
        selected_values = []  # 存储解码后的 reward 和 rtg
        # for test
        rtg_preds_ls = []
        for i in range(B):
            for k in range(K):
                best_idx = best_group_indices[i, k].item()
                group_start = context_lengths[i][k] + best_idx * (20)
                # 收集 hidden states
                for token_idx in range(self.reward_group_size):
                    critical_segments.append(h[i:i+1, group_start+token_idx*2, :])
                critical_segments.append(h[i:i+1, group_start + self.reward_group_size*2 - 1, :])
                
                # 收集对应的解码值：前 reward_group_size 个来自 reward_preds，最后1个来自 rtg_preds
                # reward_preds[i, k, best_idx]: [10, 14]
                for token_idx in range(self.reward_group_size):
                    selected_values.append(reward_preds[i, k, best_idx, token_idx, :])  # [14]
                selected_values.append(rtg_preds[i, k, best_idx, self.reward_group_size - 1, :])  # [14]
                
                rtg_preds_ls.append(rtg_preds[i, k, best_idx, :, :])  # [10, 14]
                
        critical_segments = torch.cat(critical_segments, dim=0)  # [B*K*(S+1), dim]
        critical_segments = critical_segments.view(B, K, self.reward_group_size + 1, -1)
        
        selected_values = torch.stack(selected_values, dim=0)  # [B*K*(S+1), 14]
        selected_values = selected_values.view(B, K, self.reward_group_size + 1, -1)  # [B, K, S+1, 14]

        rtg_preds_tensor = torch.cat(rtg_preds_ls, dim=0)  # [B*K*10, 14]
        rtg_preds_tensor = rtg_preds_tensor.view(B, K, 10, -1)  # [B,K,10,14]
        last_image_token_pos_tensor = torch.tensor(last_image_token_pos, device=device).view(B, K)

        # 返回奖励采样结果和模型状态
        return {
            'logits': logits,
            'reward_preds_group_mean': torch.mean(reward_preds[..., -1], dim=-1),
            'best_reward_group': best_group_indices, # [B,K]
            'selected_values': selected_values, # [B,K,reward_group_size+1,reward_dim]
            'last_hidden_states': h, # [B,max_len,H]
            'last_image_token_pos': last_image_token_pos_tensor, # [B,K](In general [B,1])
            'critical_segments': critical_segments, # [B,K,S+1,H]
            'context_lengths': context_lengths, # [B,K]
            'reward_embedding_norm': reward_scale,
            'noise_norm': avg_noise_norm,
            'rwd_noise_ratio': rwd_noise_ratio,
            'rtg_noise_ratio': rtg_noise_ratio,
            'rtg_preds':rtg_preds_tensor
        }

