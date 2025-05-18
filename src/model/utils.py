import torch

T = torch.Tensor


def expand_first(
    feat: T,
    scale=1.0,
) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


def masked_scaled_dot_product_attention(
    attn,
    query,
    key,
    value,
    attn_mask,
    fixed_token_indices=None,
    attn_scale=2.5
):
    """
    attn: attention processor
    query: (batch_size, num_heads, seq1_length, dim_head)
    key: (batch_size, num_heads, seq2_length, dim_head)
    value: (batch_size, num_heads, seq2_length, dim_head)
    attn_mask: (batch_size, seq1_length, seq2_length), inside is 1, outside is 0
    """
    logits = torch.einsum("bhqd,bhkd->bhqk", query, key) * attn.scale
    probs = logits.softmax(dim=-1) # (batch_size, num_heads, seq1_length, seq2_length)
    attn_mask = attn_mask.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
    probs = probs * attn_mask
    if fixed_token_indices is not None:
        inside_indices = torch.tensor([i for i in range(1, 39)])
        outside_indices = torch.tensor([i for i in range(39, 77)])
        probs[:, :, :, inside_indices] *= attn_scale
        probs[:, :, :, outside_indices] *= attn_scale
        probs[:, :, :, fixed_token_indices] /= attn_scale
    return torch.einsum("bhqk,bhkd->bhqd", probs, value)
