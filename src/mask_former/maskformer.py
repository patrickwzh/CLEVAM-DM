import detectron2
import torch
import numpy as np
import einops

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from src.mask_former.config import add_mask_former_config
from src.mask_former.data.datasets.register_coco_stuff_10k import COCO_CATEGORIES


def setup_cfg(cfg_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    print(f"seg cfg {cfg}")
    return cfg



def infer_maskformer(imgs, source_prompt, maskformer_model, aug):
    """
    Returns a binary mask for the specified source prompt using MaskFormer.
    """
    category_mapping = {cat["name"]: i for i, cat in enumerate(COCO_CATEGORIES)}
    if source_prompt not in category_mapping:
        raise ValueError(f"Prompt '{source_prompt}' not found in category mapping.")
    category_idx = category_mapping[source_prompt]
    # img = img.unsqueeze(0)
    # img = img.to(torch.float32) / 255.0
    # img = np.array(img)
    _, height, width, _ = imgs.shape
    print(f"shapes: {imgs.shape}")
    imgs = [np.array(img) for img in imgs]
    imgs = np.array([aug.get_transform(img).apply_image(img) for img in imgs])
    imgs = torch.as_tensor(imgs.astype("float32").transpose(0, 3, 1, 2))

    inputs = [{"image": img, "height": height, "width": width} for img in imgs]
    with torch.no_grad():
        predictions = maskformer_model(inputs)
        predictions = torch.stack([pred["sem_seg"] for pred in predictions])
        # predictions = predictions.detach().cpu().numpy()
    # print(f"preds: {predictions}")
    mask_preds = predictions.argmax(dim=1)
    mask_preds = (mask_preds == category_idx).to(torch.uint8)
    mask_preds = mask_preds.detach().cpu().numpy()
    return mask_preds
