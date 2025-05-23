from diffusers import (
    StableDiffusionBrushNetPipeline,
    BrushNetModel,
    UniPCMultistepScheduler,
)
import torch
import numpy as np
from PIL import Image
import cv2
import os
from src.model.attention_processors import Handler, StyleAlignedArgs
from src.utils import (
    get_keyframes,
    save_processed_keyframes,
)


class ConsistentLocalEdit:
    def __init__(self, cfg):
        base_model_path = os.path.join(cfg.brushnet_path, "realisticVisionV60B1_v51VAE")
        brushnet_path = os.path.join(
            cfg.brushnet_path, "segmentation_mask_brushnet_ckpt"
        )
        self.brushnet = BrushNetModel.from_pretrained(
            brushnet_path, torch_dtype=torch.float16
        )
        self.brushnet = self.brushnet
        self.pipeline = StableDiffusionBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=self.brushnet,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.pipeline.safety_checker = lambda images, clip_input: (
            images,
            [False] * len(images),
        )
        self.pipeline.enable_model_cpu_offload(device=cfg.device)
        handler = Handler(self.pipeline)
        sa_args = StyleAlignedArgs(
            share_group_norm=True,
            share_layer_norm=True,
            share_attention=True,
            adain_queries=True,
            adain_keys=True,
            adain_values=False,
            full_attention_share=True,
        )

        handler.register(
            sa_args,
        )

    def process(self, cfg):
        keyframes = get_keyframes(cfg)
        print("\tLoading segmentation masks...")
        masks = np.load(os.path.join(cfg.keyframe_path, "masks.npy"))
        print("\tSegmentation masks loaded.")
        init_images = [
            keyframe * (1 - mask) for keyframe, mask in zip(keyframes, masks)
        ]

        batch_size = len(keyframes)

        init_images = [
            Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
            for init_image in init_images
        ]
        masks = [
            Image.fromarray(mask.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")
            for mask in masks
        ]
        prompts = [cfg.prompts.edit_inside] * batch_size

        generator = torch.Generator(cfg.device).manual_seed(42)
        print("\tProcessing keyframes...")
        images = self.pipeline(
            prompts,
            init_images,
            masks,
            num_inference_steps=cfg.num_inference_steps,
            generator=generator,
            brushnet_conditioning_scale=1.0,
            guidance_scale=cfg.guidance_scale,
        ).images
        print("\tKeyframes processed. Saving images...")

        save_processed_keyframes(images, cfg)
