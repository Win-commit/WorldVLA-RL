"""
Patch for Emu3Model to support custom 4D attention masks for parallel reward sampling.
"""

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from models.Emu3.emu3.mllm.modeling_emu3 import Emu3Model


def patched_emu3_forward(self, input_ids=None, attention_mask=None, position_ids=None, 
                        past_key_values=None, inputs_embeds=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None, return_dict=None,
                        custom_4d_attention_mask=None):
    """
    Patched forward method for Emu3Model that supports custom 4D attention masks.
    
    Args:
        custom_4d_attention_mask: Optional 4D attention mask [batch, heads, seq_len, seq_len]
                                 If provided, this will be used instead of generating causal mask
        ... (other args same as original)
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        from transformers.cache_utils import Cache, DynamicCache
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # ===== MODIFIED PART: Support custom 4D attention mask =====
    if custom_4d_attention_mask is not None:
        # Use the provided custom 4D attention mask directly
        attention_mask = custom_4d_attention_mask
    else:
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
    # ===== END MODIFIED PART =====

    # embed positions
    hidden_states = self.dropout(inputs_embeds)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        from transformers.cache_utils import Cache
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def apply_emu3_parallel_patch(model: Emu3Model):
    """
    Apply the parallel reward sampling patch to an Emu3Model instance.
    
    
    Returns:
        The patched model (modified in-place)
    """
    # Replace the forward method with our patched version
    import types
    model.forward = types.MethodType(patched_emu3_forward, model)
    return model


def generate_parallel_reward_attention_mask(seq_length, context_len, reward_groups, reward_group_size, device):
    """
    Generate 4D attention mask for parallel reward sampling.
    
    Args:
        seq_length: Total sequence length
        context_len: Length of context (text + state + images)
        reward_groups: Number of reward sample groups (M)
        reward_group_size: Size of each reward group
        device: Device to create the mask on
    
    Returns:
        4D attention mask [1, 1, seq_len, seq_len] ready for use
    """
    # Start with causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
    
    # Apply parallel reward constraints
    reward_start = context_len
    for i in range(reward_groups):
        group_start = reward_start + i * reward_group_size
        group_end = group_start + reward_group_size
        
        # Block attention to other reward groups
        for j in range(reward_groups):
            if i != j:
                other_group_start = reward_start + j * reward_group_size
                other_group_end = other_group_start + reward_group_size
                
                # Block attention between different groups
                mask[group_start:group_end, other_group_start:other_group_end] = 0
    
    attention_mask = mask.masked_fill(mask == 0, -1e9)
    attention_mask = attention_mask.masked_fill(mask == 1, 0.0)
    
    # Add batch and head dimensions [1, 1, seq_len, seq_len]
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    
    return attention_mask