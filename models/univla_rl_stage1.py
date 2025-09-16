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



class Emu3RewardModel(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, tokenizer: Emu3Tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.model: Emu3Model = Emu3Model(config)
        self.hidden_dim = config.hidden_size
        self.proprio = ProprioProjector(self.hidden_dim)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reward_head = RwdHead()
        self.rtg_head = RtgHead()

        self.ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in {
            'bos': tokenizer.bos_token,
            'state_beg':tokenizer.state_beg_token,
            'state_end':tokenizer.state_end_token,
            'rwd':tokenizer.rwd_token,
            'rtg':tokenizer.rtg_token,
            'pad':tokenizer.pad_token
        }.items()}

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
        text_ids_list: list,
        image_token_ids: torch.LongTensor,
        states: torch.Tensor,
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
        label_list = []  # for CE over image tokens

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
                parts += [static['rwd']]
                labels += [-100]

            # RTG tokens
            parts += [static['rtg']]
            labels += [-100]

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
        reward_preds = []
        rtg_preds = []
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
                reward_preds.append(self.reward_head(h[i:i+1, pos, :]))
                # Move past reward token
                pos += 1
            rtg_preds.append(self.rtg_head(h[i:i+1, pos, :]))
        reward_preds = torch.stack(reward_preds, dim=1).view(B, K, -1) # [B,K,14]
        rtg_preds = torch.stack(rtg_preds, dim=0)      # [B,1,14]
        rwd_mse = 0
        rtg_mse = 0
        if reward_targets is not None:
            rwd_mse += F.mse_loss(reward_preds, reward_targets)
        if rtg_targets is not None:
            rtg_mse += F.mse_loss(rtg_preds, rtg_targets[:, -1])

        return {'loss': img_ce + rwd_mse + rtg_mse,
                'vision_loss': img_ce,
                'reward_loss': None if reward_targets is None else rwd_mse,
                'rtg_loss': None if rtg_targets is None else rtg_mse,
                'logits': logits,
                'reward_preds': reward_preds,
                'rtg_pred': rtg_preds}