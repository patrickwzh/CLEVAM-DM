import torch
import torch.nn as nn
import einops
from typing import Optional
from tqdm import tqdm
from PIL import Image
import numpy as np

from lang_sam import LangSAM

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

def get_segmentation_masks(cfg, keyframes, save_segm=True):
    keyframes = [Image.fromarray(keyframe).convert('RGB') for keyframe in keyframes]
    model = LangSAM(device=cfg.device)
    segms_all = []
    chunk_size = min(cfg.chunk_size, len(keyframes))
    with torch.no_grad():
        for i in tqdm(range(0, len(keyframes), chunk_size), desc="Processing keyframes"):
            end = min(i + chunk_size, len(keyframes))
            chunk = keyframes[i:end]
            prompts_chunk = [cfg.prompts.original_inside] * len(chunk)
            segms = model.predict(chunk, prompts_chunk)
            segms = [segm["masks"][0] for segm in segms]
            if save_segm:
                for j, segm in enumerate(segms):
                    segm = segm * 255
                    segm = segm.astype(np.uint8)
                    Image.fromarray(segm).save(f"{cfg.keyframe_path}/segm_{i + j}.png")
            segms = [np.expand_dims(segm, axis=0) for segm in segms]
            segms_all.extend(segms)
        segms_all = np.concatenate(segms_all, axis=0)
        segms_all = np.expand_dims(segms_all, axis=-1)
    np.save(f"{cfg.keyframe_path}/masks.npy", segms_all)
    return segms_all
