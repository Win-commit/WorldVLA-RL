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
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union
from torch.cuda.amp import autocast
from models.reward_heads import RwdHead,RtgHead
from models.Emu3.emu3.mllm.modeling_emu3 import Emu3PreTrainedModel,Emu3Model
from models.Emu3.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from PIL import Image
from typing import List
from models.Projectors import ProprioProjector
from models.emu3_parallel_patch import apply_emu3_parallel_patch, generate_parallel_reward_attention_mask



class Emu3ParallelRewardModel(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, tokenizer: Emu3Tokenizer, parallel_reward_groups=4, reward_group_size=10, gamma=0.9):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.model: Emu3Model = Emu3Model(config)
        self.hidden_dim = config.hidden_size
        self.proprio = ProprioProjector(self.hidden_dim)
        
        # Parallel reward sampling parameters
        self.parallel_reward_groups = parallel_reward_groups  # M groups
        self.reward_group_size = reward_group_size  # tokens per reward group (not including rtg)
        self.use_parallel_reward_sampling = True  # Always enabled for this model
        
        # noise factor
        self.noise_factor = 0.1

        # Time weighting parameter for reward/rtg aggregation
        self.gamma = gamma  # Default gamma for time weighting

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reward_head = RwdHead()
        self.rtg_head = RtgHead()

        self.ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in {
            'bos': tokenizer.bos_token,
            'state_beg': tokenizer.state_beg_token,
            'state_end': tokenizer.state_end_token,
            'rwd': tokenizer.rwd_token,
            'rwd_beg': tokenizer.rwd_beg_token,
            'rwd_end': tokenizer.rwd_end_token,
            'rtg': tokenizer.rtg_token,
            'pad': tokenizer.pad_token,
            'boa': tokenizer.boa_token,
            'eoa': tokenizer.eoa_token,
        }.items()}

        self.detach_selected_reward_hs = True

        # Apply the parallel reward sampling patch to the model
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
        action_token_ids: List[torch.Tensor] = None,
        reward_targets: torch.Tensor = None,
        rtg_targets: torch.Tensor = None,
    ):
        B, K, L_img = image_token_ids.shape
        device = image_token_ids.device
        max_len = self.tokenizer.model_max_length
        
        # prepare embeddings
        static = {k: self.get_input_embeddings()(torch.full((1,1), v, device=device)) for k,v in self.ids.items() if k!='pad'}
        pad_emb = self.get_input_embeddings()(torch.full((1,1), self.ids['pad'], device=device))
        img_embs = self.get_input_embeddings()(image_token_ids)        # [B,K,L_img,H]
        state_embs = self.proprio(states)             # [B,K,H]

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
                                # Sample group-level Gaussian noise for diversity
                group_noise = torch.randn(1, self.hidden_dim, device=device, dtype=static['rwd'].dtype) * self.noise_factor
                
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
            reward_preds.extend(sample_reward_preds)  # M predictions per sample
        
        # Reshape reward and rtg predictions: [B*M, reward_dim] -> [B, M, reward_dim]
        reward_preds = torch.stack(reward_preds, dim=0).view(B, self.parallel_reward_groups, -1)

        # 选择最大 reward 的组
        reward_scores = reward_preds.mean(dim=-1)  # [B, M]
        best_group_indices = torch.argmax(reward_scores, dim=1)  # [B]

        # 构建第二次前向（用于 action NTP），仅在提供 action_token_ids 时执行
        action_logits = None
        action_ce_loss = None
        if action_token_ids is not None:

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

                # rwd_beg
                parts2 += [static['rwd_beg']]
                labels2 += [-100]

                # 选中组的 reward hidden states（来自第一次前向的最后隐状态）
                context_len_i = context_lengths[i]
                best_idx = best_group_indices[i].item()
                group_start = context_len_i + best_idx * (self.reward_group_size + 1)  # +1 for rtg
                group_end = group_start + self.reward_group_size + 1  # +1 for rtg
                selected_reward_hs = h[i:i+1, group_start:group_end, :]  # [1, G+1, H]
                if getattr(self, "detach_selected_reward_hs", True):
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
                # 上下文：text + K*(img + 3个state标记/嵌入) + 1(rwd_beg) + (G+1)(选中组的reward+rtg hs) + 1(rwd_end)
                context2_len = L_text + K * (L_img + 3) + 1 + (self.reward_group_size + 1) + 1
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
            'logits': logits,
            'reward_preds_group_mean': reward_preds,
            'best_reward_group': best_group_indices,
            'action_logits': action_logits,
            'parallel_reward_groups': self.parallel_reward_groups,
            'action_ce_loss': action_ce_loss,
        }

    def set_parallel_reward_config(self, parallel_reward_groups: int, reward_group_size: int = 1):
        """
        Update parallel reward sampling configuration.
        
        Args:
            parallel_reward_groups: Number of parallel reward groups (M)
            reward_group_size: Number of tokens per reward group
        """
        self.parallel_reward_groups = parallel_reward_groups
        self.reward_group_size = reward_group_size
