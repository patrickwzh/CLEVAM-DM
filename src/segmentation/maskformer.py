import detectron2
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor
from mask_former.config import add_mask_former_config
from mask_former.data.datasets.register_coco_stuff_10k import COCO_CATEGORIES


def setup_cfg(cfg_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg


maskformer_cfg = setup_cfg("mask_former/configs/maskformer_R50_bs32_60k.yaml")
maskformer_model = DefaultPredictor(maskformer_cfg)


def infer_maskformer(img, source_prompt):
    """
    Returns a binary mask for the specified source prompt using MaskFormer.
    """
    category_mapping = {cat["name"]: i for i, cat in enumerate(COCO_CATEGORIES)}
    if source_prompt not in category_mapping:
        raise ValueError(f"Prompt '{source_prompt}' not found in category mapping.")
    category_idx = category_mapping[source_prompt]
    img = np.array(img)
    with torch.no_grad():
        mask_preds = maskformer_model(img)["sem_seg"]
        mask_preds = mask_preds.detach().cpu().numpy()
    mask_preds = mask_preds.argmax(axis=0)
    mask_preds = np.where(mask_preds == category_idx, 1, 0)
    return mask_preds
