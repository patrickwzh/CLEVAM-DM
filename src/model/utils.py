import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from difflib import SequenceMatcher

T = torch.Tensor

def get_clip(cfg):
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16
    )
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip = clip_model.text_model
    clip = clip.to(cfg.device)
    return clip, clip_tokenizer


def get_tokens_embeds(clip_tokenizer, clip, prompt):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True
    )
    input_ids = tokens.input_ids.to(clip.device)
    embedding = clip(input_ids).last_hidden_state
    return tokens, embedding


def compute_fixed_indices(tokens_inversion, tokens, num_tokens=38):
    first_pad = tokens["attention_mask"].sum().item() - 1
    tokens_inversion = tokens_inversion.input_ids.numpy()[0]
    tokens = tokens.input_ids.numpy()[0]
    fixed_indices = []
    for name, a0, a1, b0, b1 in SequenceMatcher(
        None, tokens_inversion, tokens
    ).get_opcodes():
        if name == "equal" and b0 < first_pad:
            b1 = min(b1, num_tokens)
            fixed_indices += list(range(b0, b1))
    return torch.tensor(fixed_indices)[1:]


def init_shared_norm(pipeline):
    def register_norm_forward(
        norm_layer: nn.GroupNorm | nn.LayerNorm,
    ) -> nn.GroupNorm | nn.LayerNorm:
        if not hasattr(norm_layer, "orig_forward"):
            setattr(norm_layer, "orig_forward", norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T:
            n = hidden_states.shape[-2]
            hidden_states = concat_first(hidden_states, dim=-2)
            hidden_states = orig_forward(hidden_states)
            return hidden_states[..., :n, :]

        norm_layer.forward = forward_

    def get_norm_layers(
        pipeline_, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]
    ):
        if isinstance(pipeline_, nn.LayerNorm):
            norm_layers_["layer"].append(pipeline_)
        elif isinstance(pipeline_, nn.GroupNorm):
            norm_layers_["group"].append(pipeline_)
        else:
            for layer in pipeline_.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {"group": [], "layer": []}
    get_norm_layers(pipeline.unet, norm_layers)
    for layer in norm_layers["layer"]:
        register_norm_forward(layer)
    for layer in norm_layers["group"]:
        register_norm_forward(layer)


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
    attn, query, key, value, attn_mask, fixed_token_indices=None, attn_scale=2.5
):
    """
    attn: attention processor
    query: (batch_size, num_heads, seq1_length, dim_head)
    key: (batch_size, num_heads, seq2_length, dim_head)
    value: (batch_size, num_heads, seq2_length, dim_head)
    attn_mask: (batch_size, seq1_length, seq2_length), inside is 1, outside is 0
    fixed_token_indices: (batch_size, num_fixed_indices)
    """
    logits = torch.einsum("bhqd,bhkd->bhqk", query, key) * attn.scale
    probs = logits.softmax(dim=-1)  # (batch_size, num_heads, seq1_length, seq2_length)
    attn_mask = attn_mask.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
    probs = probs * attn_mask
    if fixed_token_indices is not None:
        inside_indices = torch.tensor([i for i in range(1, 39)])
        outside_indices = torch.tensor([i for i in range(39, 77)])
        probs[:, :, :, inside_indices] *= attn_scale
        probs[:, :, :, outside_indices] *= attn_scale
        probs[:, :, :, fixed_token_indices] /= attn_scale
    return torch.einsum("bhqk,bhkd->bhqd", probs, value)


def inversion(x, model, scheduler, original_prompt_embeds):
    seq = scheduler.timesteps
    seq = torch.flip(seq, dims=(0,))
    b = scheduler.betas
    b = b.to(x.device)

    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        seq_iter = seq_next[1:]
        seq_next_iter = seq[1:]

        x0_preds = []
        xs = [x]
        for index, (i, j) in enumerate(zip(seq_iter, seq_next_iter)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = (1 - b).cumprod(dim=0).index_select(0, t.long())
            if next_t.sum() == -next_t.shape[0]:
                at_next = torch.ones_like(at)
            else:
                at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())

            xt = xs[-1].to(x.device)

            # set_timestep(model, 0.0)

            et = model(xt, t, encoder_hidden_states=original_prompt_embeds).sample

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))
            c2 = (1 - at_next).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et
            xs.append(xt_next.to("cpu"))

    return xs, x0_preds
