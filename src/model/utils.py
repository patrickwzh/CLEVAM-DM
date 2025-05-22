import torch
import torch.nn as nn
import einops
from typing import Optional
from tqdm import tqdm
from PIL import Image
import numpy as np

from src.mask_former.mask_former_model import MaskFormer
from src.mask_former.maskformer import setup_cfg, infer_maskformer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as transforms

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

def get_segmentation_masks(cfg, keyframes):
    maskformer_cfg = setup_cfg(cfg.maskformer_path)
    maskformer_model = build_model(maskformer_cfg).to(cfg.device)
    checkpointer = DetectionCheckpointer(maskformer_model)
    checkpointer.load(maskformer_cfg.MODEL.WEIGHTS)
    maskformer_model.eval()
    checkpointer = DetectionCheckpointer(maskformer_model)
    checkpointer.load(maskformer_cfg.MODEL.WEIGHTS)
    aug = transforms.ResizeShortestEdge(
        [maskformer_cfg.INPUT.MIN_SIZE_TEST, maskformer_cfg.INPUT.MIN_SIZE_TEST], maskformer_cfg.INPUT.MAX_SIZE_TEST
    )
    keyframes = np.array(keyframes)
    segms = infer_maskformer(keyframes, cfg.prompts.original_inside, maskformer_model, aug)
    for i, segm in enumerate(segms):
        segm = segm * 255
        segm = segm.astype(np.uint8)
        Image.fromarray(segm).save(f"{cfg.keyframe_path}/segm_{i}.png")
    segms = [np.expand_dims(segm, axis=-1) for segm in segms]
    return segms
