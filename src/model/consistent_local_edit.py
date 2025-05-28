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
from src.model.utils import inversion


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
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
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
    
    def get_inverse_latents(self, keyframes, original_prompt_embeds, vae, cfg):
        keyframes = np.array(keyframes)
        keyframes = torch.from_numpy(keyframes)
        # format keyframes
        keyframes = keyframes.permute(0, 3, 1, 2).to(cfg.device)
        # Convert BGR to RGB for keyframes (assuming keyframes shape is [N, C, H, W])
        keyframes = keyframes[:, [2, 1, 0], ...]
        keyframes = keyframes / 255.0
        init_latents = vae.encode(keyframes).latent_dist.sample() * 0.18215
        batch_size, _, _, _ = init_latents.shape
        init_latents = init_latents.repeat(2, 1, 1, 1)
        latents = []
        for start in range(0, 2 * batch_size, cfg.chunk_size):
            end = min(start + cfg.chunk_size, 2 * batch_size)
            latents_i, _ = inversion(
                init_latents[start:end], self.pipeline.unet, self.pipeline.scheduler, original_prompt_embeds, cfg
            )
            latents_i = [latent_i.to(cfg.device) for latent_i in latents_i]
            latents_i = torch.stack(latents_i)
            latents.append(latents_i)
        latents = torch.stack(latents, dim=1)
        latents = latents[:, :batch_size]
        # pipeline's `prepare_latents` has scaled the latents by `init_noise_sigma`, so we need to pre-scale it back
        return latents

    
    def process(self, cfg):
        keyframes = get_keyframes(cfg)
        print("Loading segmentation masks...")
        masks = np.load(os.path.join(cfg.keyframe_path, "masks.npy"))
        print(f"mask shape: {masks.shape}")
        if cfg.change_background:
            masks = 1 - masks
        print("Segmentation masks loaded.")

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=cfg.prompts.original_inside if not cfg.change_background else cfg.prompts.original_outside,
            device=cfg.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_embeds.repeat(2 * len(keyframes), 1, 1)
        if cfg.do_inverse:
            init_latents = self.get_inverse_latents(keyframes, prompt_embeds, self.pipeline.vae, cfg)


        init_images = [
            keyframe * (1 - mask) for keyframe, mask in zip(keyframes, masks)
        ]
        # init_images = keyframes

        batch_size = len(keyframes)

        init_images = [
            Image.fromarray((init_image[:, :, ::-1]).astype(np.uint8))
            for init_image in init_images
        ]
        masks = [
            Image.fromarray(mask.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")
            for mask in masks
        ]
        print(f"len: {len(init_images)}, {len(masks)}")
        prompts = [cfg.prompts.edit_inside if not cfg.change_background else cfg.prompts.edit_outside] * batch_size

        generator = torch.Generator(cfg.device).manual_seed(42)
        print("Processing keyframes...")
        images = self.pipeline(
            prompts,
            init_images,
            masks,
            num_inference_steps=cfg.num_inference_steps,
            generator=generator,
            latents=init_latents[-1] / self.pipeline.scheduler.init_noise_sigma if cfg.do_inverse else None,
            brushnet_conditioning_scale=1.0,
            guidance_scale=cfg.guidance_scale,
        ).images
        print("Keyframes processed. Saving images...")

        save_processed_keyframes(images, cfg, RGB2BGR=True)
