import torch
import torch.nn as nn
from diffusers.models import attention_processor


class SharedCrossAttentionProcessor(nn.Module):
    def __init__(self, cross_attn_mask, fixed_token_indices):
        super().__init__()
        self.cross_attn_mask = cross_attn_mask
        self.fixed_token_indices = fixed_token_indices

    def __call__(
        self,
        attn: attention_processor.Attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        **kwargs
    ):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)