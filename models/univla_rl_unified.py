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
import random 
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, List
from torch.cuda.amp import autocast
from models.reward_heads import RwdHead, RtgHead
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
                 p: float = 0.85,
                 gamma: float = 0.9,
                 noise_factor: float = 0.6,
                 detach_selected_reward_hs: bool = True):
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
            detach_selected_reward_hs: 是否在第二次前向中分离选定的reward隐藏状态
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
        self.use_parallel_reward_sampling = parallel_mode
        self.p = p


        # 噪声强度
        self.noise_factor = noise_factor

        # 时间加权参数
        self.gamma = gamma
        
        # 是否分离选定的reward隐藏状态
        self.detach_selected_reward_hs = detach_selected_reward_hs

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reward_head = RwdHead()
        self.rtg_head = RtgHead()
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
            return self._forward_stage1(text_ids_list, image_token_ids, states, reward_targets, rtg_targets)
        else:
            return self._forward_stage2(text_ids_list, image_token_ids, states, action_token_ids, reward_targets, rtg_targets)

    def _forward_stage1(self,
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

        reward_preds = self.reward_head(reward_vectors).view(B, K*action_frames, -1)
        rtg_preds = self.rtg_head(rtg_vectors).view(B, K*action_frames, -1)
        
        rwd_mse = 0
        rtg_mse = 0
        if reward_targets is not None:
            reward_targets =  reward_targets.contiguous().view(B, K * action_frames, -1)
            rwd_mse += F.mse_loss(reward_preds, reward_targets)
        if rtg_targets is not None:
            rtg_targets = rtg_targets.contiguous().view(B, K * action_frames, -1)
            rtg_mse += F.mse_loss(rtg_preds, rtg_targets)
        # 使用EMA对不同尺度的损失进行自适应加权
        if self.auto_balance_stage1:
            eps = 1e-8
            if self.training:
                with torch.no_grad():
                    self.img_ce_ema = self.img_ce_ema * self.loss_ema_decay + (1 - self.loss_ema_decay) * img_ce.detach()
                    if reward_targets is not None:
                        self.rwd_mse_ema = self.rwd_mse_ema * self.loss_ema_decay + (1 - self.loss_ema_decay) * rwd_mse.detach()
                    if rtg_targets is not None:
                        self.rtg_mse_ema = self.rtg_mse_ema * self.loss_ema_decay + (1 - self.loss_ema_decay) * rtg_mse.detach()
            total_loss = img_ce
            if reward_targets is not None:
                rwd_w = torch.clamp(self.img_ce_ema.detach() / (self.rwd_mse_ema.detach() + eps), 0.1, 10.0)
                total_loss = total_loss + rwd_w * rwd_mse
            if rtg_targets is not None:
                rtg_w = torch.clamp(self.img_ce_ema.detach() / (self.rtg_mse_ema.detach() + eps), 0.1, 10.0)
                total_loss = total_loss + rtg_w * rtg_mse
        else:
            total_loss = img_ce + rwd_mse + rtg_mse
        
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
                'reward_preds': reward_preds,
                'rtg_pred': rtg_preds}

    def sample_rewards(self,
        text_ids_list: List,
        image_token_ids: torch.LongTensor,
        states: torch.Tensor
    ):
        """
        阶段2.1: 并行奖励采样部分，仅负责生成并行奖励组并选择最佳奖励组
        
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

        for i, text_ids in enumerate(text_ids_list):
            L_text = text_ids.size(0)
            # text embeddings and initial labels
            t_emb = self.get_input_embeddings()(text_ids.unsqueeze(0))  # [1,L_text,H]
            parts = [t_emb]

            # per-frame segments (assuming K=1 for parallel reward sampling)
            for j in range(K):
                # image
                ids_ij = image_token_ids[i, j].tolist()
                emb_ij = img_embs[i:i+1, j]               # [1,L_img,H]
                parts.append(emb_ij)
                # state_beg, state, state_end
                parts += [static['state_beg'], state_embs[i:i+1,j:j+1,:], static['state_end']]
            parts += [static['rwd_beg']]
            # Calculate context length (text + images + states)
            # Each group now has reward_group_size + 1 tokens (including rtg)
            context_len = L_text + K * (L_img + 3) + 1  # 3 for state_beg, state, state_end, 1 for rwd_beg
            context_lengths.append(context_len)
            
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
                
                # Add reward tokens with noise
                for token_idx in range(self.reward_group_size):
                    rwd_emb = static['rwd'] + group_noise
                    parts += [rwd_emb]
                
                # Add rtg token at the end of each group with noise
                rtg_emb = static['rtg'] + group_noise
                parts += [rtg_emb]
            
            parts += [static['rwd_end']]

            # concat and pad to max_len
            seq_i = torch.cat(parts, dim=1)  # [1, L_i, H]
            
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
            context_len = context_lengths[i]
            sample_mask = generate_parallel_reward_attention_mask(
                seq_length=seq_len,
                context_len=context_len,
                reward_groups=self.parallel_reward_groups,
                reward_group_size=self.reward_group_size + 1,
                device=device
            )
            
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
            return_dict=True
        )
        h = outputs.last_hidden_state                 # [B, max_len, H]
        logits = self.lm_head(h)
        
        # reward/rtg preds with parallel sampling
        reward_preds = []
        
        for i, text_ids in enumerate(text_ids_list):
            pos = context_lengths[i]
            
            # Parallel reward sampling: predict from all reward groups
            sample_reward_preds = []
            
            for group_idx in range(self.parallel_reward_groups):
                group_start_pos = context_lengths[i] + group_idx * (self.reward_group_size + 1)  # +1 for rtg
                
                # Collect all tokens in this group (rewards + rtg) for time-weighted aggregation
                group_tokens = []
                group_weights = []
                
                # Decode reward tokens
                for token_idx in range(self.reward_group_size):
                    current_pos = group_start_pos + token_idx
                    hidden_state = h[i:i+1, current_pos, :]
                    reward_pred = self.reward_head(hidden_state)
                    group_tokens.append(reward_pred)
                    group_weights.append(torch.pow(torch.tensor(self.gamma, device=h.device), token_idx + 1))  # gamma^1, gamma^2, ..., gamma^group_size
                
                # Decode rtg token (last token in the group)
                rtg_pos = group_start_pos + self.reward_group_size
                rtg_hidden_state = h[i:i+1, rtg_pos, :]
                rtg_pred = self.rtg_head(rtg_hidden_state)
                group_tokens.append(rtg_pred)
                group_weights.append(torch.pow(torch.tensor(self.gamma, device=h.device), self.reward_group_size + 1))  # gamma^(group_size+1)
                
                # Time-weighted aggregation: gamma^1*rwd1 + gamma^2*rwd2 + ... + gamma^(group_size+1)*rtg
                # Stack all predictions and apply time weights
                tokens_stack = torch.stack(group_tokens, dim=0)  # [group_size+1, 1, dim]
                weights_tensor = torch.tensor(group_weights, device=h.device, dtype=tokens_stack.dtype).unsqueeze(1).unsqueeze(2)  # [group_size+1, 1, 1]
                weighted_tokens = tokens_stack * weights_tensor
                group_total = weighted_tokens.sum(dim=0)  # [1, dim]

                sample_reward_preds.append(group_total)
            
            # Stack all group predictions for this sample
            reward_preds.extend(sample_reward_preds)  # M(group nums) predictions per sample
        
        # Reshape reward and rtg predictions: [B*M, reward_dim] -> [B, M, reward_dim]
        reward_preds = torch.stack(reward_preds, dim=0).view(B, self.parallel_reward_groups, -1)

        # 选择最大 reward 的组
        reward_scores = reward_preds.mean(dim=-1)  # [B, M]
        best_group_indices = torch.argmax(reward_scores, dim=1)  # [B]
        if self.detach_selected_reward_hs:
            h = h.detach()
            
        # 计算平均噪声强度
        avg_noise_norm = sum(all_noise_norms) / len(all_noise_norms) if all_noise_norms else 0
        rwd_noise_ratio = avg_noise_norm / reward_scale if reward_scale > 0 else 0
        rtg_noise_ratio = avg_noise_norm / rtg_scale if rtg_scale > 0 else 0
        
        # 返回奖励采样结果和模型状态
        return {
            'logits': logits,
            'reward_preds_group_mean': reward_preds,
            'best_reward_group': best_group_indices,
            'hidden_states': h,
            'context_lengths': context_lengths,
            'reward_embedding_norm': reward_scale,
            'noise_norm': avg_noise_norm,
            'rwd_noise_ratio': rwd_noise_ratio,
            'rtg_noise_ratio': rtg_noise_ratio
        }


    def generate_actions(self,
        text_ids_list: List,
        image_token_ids: torch.LongTensor,
        states: torch.Tensor,
        action_token_ids: List[torch.Tensor],
        reward_sampling_results: Optional[Dict] = None
    ):
        """
        阶段2.2: 动作生成部分，使用最佳奖励组生成动作序列
        
        Args:
            text_ids_list: 文本ID列表
            image_token_ids: 图像token IDs
            states: 机器人状态
            action_token_ids: 动作token IDs (用于训练/评估)
            reward_sampling_results: 奖励采样结果，包含隐藏状态、最佳奖励组等信息，如果为None则不使用reward
            
        Returns:
            dict: 包含动作logits和损失等
        """
        B, K, L_img = image_token_ids.shape
        device = image_token_ids.device
        max_len = self.tokenizer.model_max_length
        
        # prepare embeddings
        static = {k: self.get_input_embeddings()(torch.full((1,1), v, device=device)) for k,v in self.ids.items() if k!='pad'}
        pad_emb = self.get_input_embeddings()(torch.full((1,1), self.ids['pad'], device=device))
        img_embs = self.get_input_embeddings()(image_token_ids)        # [B,K,L_img,H]
        state_embs = self.proprio(states)             # [B,K,H]
        
        # 检查是否有reward信息
        has_reward = reward_sampling_results is not None
        
        # 如果有reward_sampling_results，获取必要信息
        h = None
        context_lengths = None
        best_group_indices = None
        
        if has_reward:
            h = reward_sampling_results['hidden_states']
            context_lengths = reward_sampling_results['context_lengths']
            best_group_indices = reward_sampling_results['best_reward_group']
        
        seq2_list = []
        mask2_list = []
        action_labels_list = []  # 对应每个样本的完整labels（包括-100填充）
        action_pos_ranges = []  # 记录每个样本(action_start, action_len)

        for i, text_ids in enumerate(text_ids_list):
            L_text = text_ids.size(0)
            # 文本嵌入
            t_emb = self.get_input_embeddings()(text_ids.unsqueeze(0))  # [1,L_text,H]
            parts2 = [t_emb]
            labels2 = [-100] * L_text  

            # per-frame 部分
            for j in range(K):
                parts2.append(img_embs[i:i+1, j])
                labels2 += [-100] * L_img  # image部分不计算loss
                parts2 += [static['state_beg'], state_embs[i:i+1, j:j+1, :], static['state_end']]
                labels2 += [-100, -100, -100]  # state标记不计算loss

            # 如果有reward信息，添加reward相关token
            if has_reward and context_lengths is not None and best_group_indices is not None and h is not None:
                # rwd_beg
                parts2 += [static['rwd_beg']]
                labels2 += [-100]

                # 选中组的 reward hidden states（来自第一次前向的最后隐状态）
                context_len_i = context_lengths[i]
                best_idx = best_group_indices[i].item()
                group_start = context_len_i + best_idx * (self.reward_group_size + 1)  # +1 for rtg
                group_end = group_start + self.reward_group_size + 1  # +1 for rtg
                selected_reward_hs = h[i:i+1, group_start:group_end, :]  # [1, G+1, H]
                if self.detach_selected_reward_hs:
                    selected_reward_hs = selected_reward_hs.detach()
                parts2.append(selected_reward_hs)
                labels2 += [-100] * (self.reward_group_size + 1)  # reward + rtg部分不计算loss
                parts2.append(static['rwd_end'])
                labels2 += [-100]
            
            # 追加 action chunk：boa + action_ids + eoa
            boa_emb = static['boa']
            eoa_emb = static['eoa']
            parts2.append(boa_emb)
            labels2 += [-100]  # boa不计算loss
            
            act_ids_i = action_token_ids[i][0].to(device).view(1, -1)
            act_len_i = act_ids_i.size(1)
            act_emb_i = self.get_input_embeddings()(act_ids_i)  # [1, T, H]
            parts2.append(act_emb_i)
            labels2 += act_ids_i.view(-1).tolist()  # action部分计算loss
            parts2.append(eoa_emb)
            labels2 += [-100]  # eoa不计算loss

            # 计算第二次序列中 action logits 的起始位置与长度
            if has_reward:
                # 上下文：text + K*(img + 3个state标记/嵌入) + 1(rwd_beg) + (G+1)(选中组的reward+rtg hs) + 1(rwd_end)
                context2_len = L_text + K * (L_img + 3) + 1 + (self.reward_group_size + 1) + 1
            else:
                # 上下文：text + K*(img + 3个state标记/嵌入) 
                context2_len = L_text + K * (L_img + 3)
                
            boa_pos = context2_len
            action_start = boa_pos  # 对应预测第一个 action token 的 logits 起点
            action_len = act_len_i

            # 拼接为单条序列
            seq2_i = torch.cat(parts2, dim=1)  # [1, L2_i, H]
            labels2_i = torch.tensor(labels2, device=device, dtype=torch.long)

            # padding 或 截断
            L2_i = seq2_i.size(1)
            if L2_i < max_len:
                pad_len = max_len - L2_i
                seq2_i = torch.cat([seq2_i, pad_emb.expand(1, pad_len, self.hidden_dim)], dim=1)
                labels2_i = torch.cat([labels2_i, torch.full((pad_len,), -100, device=device)], dim=0)
                mask2_i = torch.cat([torch.ones(L2_i, device=device), torch.zeros(pad_len, device=device)], dim=0)
            else:
                seq2_i = seq2_i[:, :max_len, :]
                labels2_i = labels2_i[:max_len]
                mask2_i = torch.ones(max_len, device=device)
                valid_len = max_len
                if action_start + action_len > valid_len:
                    action_len = max(0, valid_len - action_start)

            seq2_list.append(seq2_i)
            mask2_list.append(mask2_i.unsqueeze(0))
            action_labels_list.append(labels2_i.unsqueeze(0))
            action_pos_ranges.append((action_start, action_len))

        # batch 合并并进行第二次前向
        inputs_embeds_2 = torch.cat(seq2_list, dim=0)  # [B, max_len, H]
        attention_mask_2 = torch.cat(mask2_list, dim=0)  # [B, max_len]
        action_labels = torch.cat(action_labels_list, dim=0)  # [B, max_len]

        outputs2 = self.model(
            inputs_embeds=inputs_embeds_2,
            attention_mask=attention_mask_2,
            return_dict=True,
        )
        h2 = outputs2.last_hidden_state
        logits2 = self.lm_head(h2)  # [B, max_len, V]

        shift_logits2 = logits2[..., :-1, :].contiguous()
        shift_labels2 = action_labels[..., 1:].contiguous()
        action_ce_loss = F.cross_entropy(
            shift_logits2.view(-1, shift_logits2.size(-1)), 
            shift_labels2.view(-1), 
            ignore_index=-100
        )

        per_sample_logits = []
        for bi in range(B):
            start_i, len_i = action_pos_ranges[bi]
            per_sample_logits.append(logits2[bi, start_i:start_i+len_i, :])  # [len_i, V]
        action_logits = per_sample_logits  # list，每个元素shape: [T_i, V]

        return {
            'action_logits': action_logits,
            'action_ce_loss': action_ce_loss,
        }

    def generate_actions_inference(self,
        text_ids_list,
        image_token_ids,
        states,
        reward_sampling_results: Optional[Dict] = None,
        action_tokenizer=None,
        max_new_tokens=80,
        action_vocab_size=2048,
        action_dim=7,
        time_horizon=10,
        do_sample=False,
        auto_sample_reward=True
    ):
        """
        用于推理时生成动作序列
        
        Args:
            text_ids_list: 文本token IDs列表
            image_token_ids: 图像token IDs
            states: 机器人状态
            reward_sampling_results: 奖励采样结果
            action_tokenizer: 动作解码器
            max_new_tokens: 生成的最大token数量
            action_vocab_size: 动作token词表大小
            action_dim: 动作维度
            time_horizon: 生成的动作时间步长
            do_sample: 是否使用采样策略生成
            auto_sample_reward: 是否在reward_sampling_results为None时自动采样奖励
            
        Returns:
            dict: 包含生成的动作和相关信息
        """
        device = image_token_ids.device
        
        # 如果没有提供reward_sampling_results且auto_sample_reward为True，则自动调用sample_rewards
        if reward_sampling_results is None and auto_sample_reward:
            reward_sampling_results = self.sample_rewards(
                text_ids_list=text_ids_list,
                image_token_ids=image_token_ids,
                states=states
            )
        
        # 检查reward_sampling_results的有效性（如果有）
        if reward_sampling_results is not None and ("best_reward_group" not in reward_sampling_results or "hidden_states" not in reward_sampling_results):
            raise ValueError("Invalid reward_sampling_results provided")
        
        # 构建前缀序列
        prefix_inputs = self._prepare_action_prefix(text_ids_list, image_token_ids, states, reward_sampling_results)
        
        # 设置生成配置
        eoa_token_id = self.ids.get('eoa', 151845)  # eoa token id
        last_token_id = self.tokenizer.pad_token_id - 1  # action token起始位置
        
        # 准备action token约束
        allowed_token_ids = list(range(last_token_id - action_vocab_size, last_token_id + 1)) + [eoa_token_id]
        
        # 创建logits处理器
        class ActionIDConstraintLogitsProcessor(LogitsProcessor):
            def __init__(self, allowed_token_ids):
                self.allowed_token_ids = allowed_token_ids

            def __call__(self, input_ids, scores):
                mask = torch.zeros_like(scores, dtype=torch.bool)
                if mask.ndim == 1:
                    mask[self.allowed_token_ids] = True
                else:
                    mask[:, self.allowed_token_ids] = True
                scores[~mask] = -float("inf")
                return scores
                
        action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
        
        # 使用自定义的generate方法进行生成
        with torch.no_grad():
            outputs = self.custom_generate(
                inputs_embeds=prefix_inputs['inputs_embeds'],
                attention_mask=prefix_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                eos_token_id=eoa_token_id,
                pad_token_id=self.config.pad_token_id,
                do_sample=do_sample,
                logits_processor=action_id_processor,
                return_dict_in_generate=True,
                output_scores=False
            )

        # 只取新生成的部分（去掉前缀）
        prefix_len = prefix_inputs['inputs_embeds'].shape[1]
        action_ids = outputs["sequences"][:, 1:-1]  # 去掉首尾的填充和eos
        
        # 处理action ids得到实际动作值
        last_token_id_tensor = torch.tensor(last_token_id, dtype=action_ids.dtype, device=device)
        processed_outputs = last_token_id_tensor - action_ids
        
        # 如果提供了action_tokenizer，则解码得到具体动作
        actions = None
        if action_tokenizer is not None:
            # 解码action tokens
            action_outputs = action_tokenizer.decode(processed_outputs, time_horizon=time_horizon, action_dim=action_dim)
            actions = action_outputs[0]  # 取第一个样本的动作序列
        
        result = {
            'action_ids': processed_outputs,
            'actions': actions,
        }
        
        # 如果有奖励采样结果，添加到返回值中
        if reward_sampling_results is not None:
            result.update({
                'best_reward_group': reward_sampling_results['best_reward_group'],
                'reward_preds_group_mean': reward_sampling_results.get('reward_preds_group_mean', None)
            })
            
        return result

    def _forward_stage2(self,
        text_ids_list: List,
        image_token_ids: torch.LongTensor,
        states: torch.Tensor,
        action_token_ids: Optional[List[torch.Tensor]] = None,
        reward_targets: Optional[torch.Tensor] = None,
        rtg_targets: Optional[torch.Tensor] = None,
    ):
        """
        Stage2前向传播（并行奖励采样）
        (荒废)
        Args:
            text_ids_list: 文本ID列表
            image_token_ids: 图像token IDs
            states: 机器人状态
            action_token_ids: 动作token IDs (用于训练/评估)
            reward_targets: 奖励目标
            rtg_targets: RTG目标
        """
        reward_sampling_results = None
        if torch.rand(1).item() < self.p:
            reward_sampling_results = self.sample_rewards(
                text_ids_list=text_ids_list,
                image_token_ids=image_token_ids,
                states=states
            )
        
        # 第二步：如果提供了action_token_ids，则进行动作生成
        action_logits = None
        action_ce_loss = None
        
        if action_token_ids is not None:
            action_generation_results = self.generate_actions(
                text_ids_list=text_ids_list,
                image_token_ids=image_token_ids,
                states=states,
                action_token_ids=action_token_ids,
                reward_sampling_results=reward_sampling_results
            )
            
            action_logits = action_generation_results['action_logits']
            action_ce_loss = action_generation_results['action_ce_loss']
        
        # 合并结果并返回
        result = {
            'action_logits': action_logits,
            'action_ce_loss': action_ce_loss,
        }
        
        # 如果有奖励采样结果，则添加到返回结果中
        if reward_sampling_results is not None:
            result.update({
                'reward_preds_group_mean': reward_sampling_results['reward_preds_group_mean'],
                'best_reward_group': reward_sampling_results['best_reward_group'],
                'noise_norm': reward_sampling_results['noise_norm'],
                'rwd_noise_ratio': reward_sampling_results['rwd_noise_ratio'],
                'rtg_noise_ratio': reward_sampling_results['rtg_noise_ratio'],
                'reward_embedding_norm': reward_sampling_results['reward_embedding_norm'],
            })
        
        return result

    def set_mode(self, parallel_mode: bool = None):
        """
        Args:
            parallel_mode: 如果为True，使用并行reward采样模式（Stage2）；如果为False，使用单个reward模式（Stage1）
        """
        if parallel_mode is not None:
            old_mode = self.parallel_mode
            self.parallel_mode = parallel_mode
            
            # 如果从非并行模式切换到并行模式，应用并行奖励采样补丁
            if not old_mode and parallel_mode:
                apply_emu3_parallel_patch(self.model)
    
    def set_parallel_reward_config(self, parallel_reward_groups: int = None, reward_group_size: int = None):
        """
        Args:
            parallel_reward_groups: 并行奖励组数（M）
            reward_group_size: 每组奖励的token数量
        """
        if parallel_reward_groups is not None:
            self.parallel_reward_groups = parallel_reward_groups
        if reward_group_size is not None:
            self.reward_group_size = reward_group_size 


    def _prepare_action_prefix(self, text_ids_list, image_token_ids, states, reward_sampling_results):
        """
        构建完整的前缀序列，包括文本、图像、状态和最佳奖励组
        
        Args:
            text_ids_list: 文本token IDs
            image_token_ids: 图像token IDs
            states: 机器人状态
            reward_sampling_results: 奖励采样结果，如果为None则不使用reward
        
        Returns:
            prefix_inputs: 用于生成的前缀输入，包含inputs_embeds和attention_mask
        """
        device = image_token_ids.device
        
        # 获取模型内部的嵌入
        static = {k: self.get_input_embeddings()(torch.full((1,1), v, device=device)) 
                for k,v in self.ids.items() if k!='pad'}
        pad_emb = self.get_input_embeddings()(torch.full((1,1), self.ids['pad'], device=device))
        img_embs = self.get_input_embeddings()(image_token_ids)
        state_embs = self.proprio(states)
        
        # 检查是否有reward信息
        has_reward = reward_sampling_results is not None
        
        # 构建前缀序列
        prefix_parts = []
        
        # 1. 文本部分
        text_ids = text_ids_list[0]
        t_emb = self.get_input_embeddings()(text_ids.unsqueeze(0))
        prefix_parts.append(t_emb)
        
        # 2. 图像和状态部分
        K = image_token_ids.shape[1]  # 图像数量
        for j in range(K):
            prefix_parts.append(img_embs[0:1, j])
            prefix_parts += [static['state_beg'], state_embs[0:1,j:j+1,:], static['state_end']]
        
        # 3. 如果有奖励信息，添加奖励部分
        if has_reward:
            # 奖励部分开始
            prefix_parts.append(static['rwd_beg'])
            
            # 选择的最佳奖励组
            h = reward_sampling_results['hidden_states']
            context_lengths = reward_sampling_results['context_lengths']
            best_group_indices = reward_sampling_results['best_reward_group']
            
            best_idx = best_group_indices[0].item()
            context_len = context_lengths[0]
            group_start = context_len + best_idx * (self.reward_group_size + 1)  # +1 for rtg
            group_end = group_start + self.reward_group_size + 1  # +1 for rtg
            selected_reward_hs = h[0:1, group_start:group_end, :]
            
            # 添加奖励隐状态
            prefix_parts.append(selected_reward_hs)
            
            # 奖励部分结束
            prefix_parts.append(static['rwd_end'])
        
        # 4. 添加动作开始标记
        prefix_parts.append(static['boa'])
        
        # 拼接为单个序列
        prefix_embeds = torch.cat(prefix_parts, dim=1)
        
        # 构建attention_mask
        prefix_len = prefix_embeds.size(1)
        attention_mask = torch.ones((1, prefix_len), dtype=torch.long, device=device)
        
        return {
            'inputs_embeds': prefix_embeds,
            'attention_mask': attention_mask
        }
        
    def custom_generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        max_new_tokens: int = 80,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logits_processor: Optional[List[LogitsProcessor]] = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        **model_kwargs
    ):
        """
        基于embeddings的自定义生成方法
        
        Args:
            inputs_embeds: 输入的embeddings序列 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            max_new_tokens: 最大生成token数
            eos_token_id: 结束标记ID
            pad_token_id: 填充标记ID
            do_sample: 是否使用采样生成
            temperature: 采样温度
            top_p: 采样的累积概率阈值
            logits_processor: logits处理器或处理器列表
            return_dict_in_generate: 是否返回dict形式的输出
            output_scores: 是否输出生成分数
        """
        # 处理默认参数
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.ids.get('eoa', 151845)  # 使用eoa作为结束符
        
        # 处理logits处理器
        if logits_processor is None:
            logits_processor = []
        elif not isinstance(logits_processor, list):
            logits_processor = [logits_processor]
        
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        vocab_size = self.lm_head.out_features
        hidden_dim = inputs_embeds.shape[2]
        
        # 初始化结果存储
        input_ids = torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device)  # 用于记录生成的token ids
        all_token_ids = input_ids.clone()  # 存储所有生成的token IDs，包括初始填充token
        scores = [] if output_scores else None
        
        # 获取word embeddings矩阵，用于后续token->embedding转换
        word_embeddings = self.get_input_embeddings().weight
        
        # 迭代生成max_new_tokens个token
        for i in range(max_new_tokens):
            model_inputs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            hidden_states = outputs.last_hidden_state
            
            next_token_logits = self.lm_head(hidden_states[:, -1, :])
            
            for processor in logits_processor:
                next_token_logits = processor(input_ids, next_token_logits)
            
            if do_sample:
                next_token_logits = next_token_logits / temperature
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除概率累积值超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 设置被移除token的logits为-inf
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float("inf")
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            if output_scores:
                scores.append(next_token_logits)
            
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            all_token_ids = torch.cat([all_token_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            token_embeds = torch.index_select(word_embeddings, 0, next_tokens).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            inputs_embeds = torch.cat([inputs_embeds, token_embeds], dim=1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                ], dim=1)
            
            if eos_token_id is not None and torch.any(next_tokens == eos_token_id):
                if torch.all(next_tokens == eos_token_id):
                    break
            
        # 处理输出
        if return_dict_in_generate:
            return {
                "sequences": all_token_ids,  # 包含初始填充token
                "scores": scores if output_scores else None
            }
        else:
            return all_token_ids 

    