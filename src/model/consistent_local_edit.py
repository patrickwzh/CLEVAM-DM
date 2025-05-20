import torch
from src.model.attention_processors import (
    SharedSelfAttentionProcessor,
    SharedCrossAttentionProcessor,
)
from src.model.utils import (
    init_shared_norm,
    get_clip,
    inversion,
    get_tokens_embeds,
    compute_fixed_indices,
    concat_first,
)
from diffusers import StableDiffusionPipeline, DDIMScheduler
from src.utils import get_keyframes, save_processed_keyframes
from src.segmentation.maskformer import infer_maskformer, setup_cfg
# from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from src.mask_former.mask_former_model import MaskFormer
import detectron2.data.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm


class ConsistentLocalEdit:
    def __init__(self, cfg):
        self.clip, self.clip_tokenizer = get_clip(cfg)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="fp16", scheduler=self.scheduler
        )
        self.pipeline = self.pipeline.to(cfg.device)

    def get_inverse_latents(self, keyframes, original_prompt_embeds, vae):
        # format keyframes
        keyframes = keyframes.permute(0, 3, 1, 2).to(self.pipeline.device)
        # Convert BGR to RGB for keyframes (assuming keyframes shape is [N, C, H, W])
        keyframes = keyframes[:, [2, 1, 0], ...]
        keyframes = keyframes / 255.0
        init_latents = vae.encode(keyframes).latent_dist.sample() * 0.18215
        latents, _ = inversion(
            init_latents, self.pipeline.unet, self.scheduler, original_prompt_embeds
        )
        # pipeline's `prepare_latents` has scaled the latents by `init_noise_sigma`, so we need to pre-scale it back
        return latents[-1] / self.pipeline.scheduler.init_noise_sigma

    def process_prompts(self, cfg):
        _, embedding_temp = get_tokens_embeds(self.clip_tokenizer, self.clip, "", cfg)
        prompt_collect_indices = torch.tensor([i for i in range(1, 39)])
        prompt_inside_indices = torch.tensor([i for i in range(1, 39)])
        prompt_outside_indices = torch.tensor([i for i in range(39, 77)])

        original_prompt_embeds = embedding_temp.clone()
        original_inside_tokens, original_inside_embeds = get_tokens_embeds(
            self.clip_tokenizer, self.clip, cfg.prompts.original_inside, cfg
        )
        original_outside_tokens, original_outside_embeds = get_tokens_embeds(
            self.clip_tokenizer, self.clip, cfg.prompts.original_outside, cfg
        )
        original_prompt_embeds[:, prompt_inside_indices, :] = original_inside_embeds[
            :, prompt_collect_indices, :
        ]
        original_prompt_embeds[:, prompt_outside_indices, :] = original_outside_embeds[
            :, prompt_collect_indices, :
        ]

        edit_prompt_embeds = embedding_temp.clone()
        edit_inside_tokens, edit_inside_embeds = get_tokens_embeds(
            self.clip_tokenizer, self.clip, cfg.prompts.edit_inside, cfg
        )
        edit_outside_tokens, edit_outside_embeds = get_tokens_embeds(
            self.clip_tokenizer, self.clip, cfg.prompts.edit_outside, cfg
        )
        edit_prompt_embeds[:, prompt_inside_indices, :] = edit_inside_embeds[
            :, prompt_collect_indices, :
        ]
        edit_prompt_embeds[:, prompt_outside_indices, :] = edit_outside_embeds[
            :, prompt_collect_indices, :
        ]

        fixed_inside_indices = compute_fixed_indices(
            original_inside_tokens, edit_inside_tokens
        )
        fixed_outside_indices = compute_fixed_indices(
            original_outside_tokens, edit_outside_tokens
        )
        fixed_outside_indices = prompt_outside_indices[fixed_outside_indices - 1]
        fixed_token_indices = torch.cat(
            (fixed_inside_indices, fixed_outside_indices), dim=0
        )

        return original_prompt_embeds, edit_prompt_embeds, fixed_token_indices
    
    def get_self_attn_mask(self, segm):
        _, n = segm.shape
        segm_ = concat_first(segm, dim=-1)
        # print(f"self_attn_mask shape: {segm_.shape}")
        mask = (segm_.unsqueeze(1) == segm_.unsqueeze(2)).to(torch.uint8)
        mask = mask[:, :n]
        # print(f"mask shape: {mask.shape}, segm_ shape: {segm_.shape}")
        return mask
    
    def get_cross_attn_mask(self, segm):
        mask = torch.zeros((*segm.shape, 77), dtype=torch.uint8).to(segm.device)
        # print(f"mask shape: {mask.shape}, segm shape: {segm.shape}, segm==1, {(segm==1).shape}")
        idx0 = torch.where(segm == 0)
        idx1 = torch.where(segm == 1)
        mask[idx1[0], idx1[1], 1:39] = 1
        mask[idx0[0], idx0[1], 39:77] = 1
        return mask
    
    def prepare_maskformer_model(self, cfg):
        maskformer_cfg = setup_cfg(cfg.maskformer_path)
        maskformer_model = build_model(maskformer_cfg).to(cfg.device)
        checkpointer = DetectionCheckpointer(maskformer_model)
        checkpointer.load(maskformer_cfg.MODEL.WEIGHTS)
        maskformer_model.eval()
        checkpointer = DetectionCheckpointer(maskformer_model)
        checkpointer.load(maskformer_cfg.MODEL.WEIGHTS)
        aug = T.ResizeShortestEdge(
            [maskformer_cfg.INPUT.MIN_SIZE_TEST, maskformer_cfg.INPUT.MIN_SIZE_TEST], maskformer_cfg.INPUT.MAX_SIZE_TEST
        )
        return maskformer_model, aug
    
    def prepare_segms(self, cfg, keyframes):
        maskformer_model, aug = self.prepare_maskformer_model(cfg)

        segms = infer_maskformer(keyframes, cfg.prompts.original_inside, maskformer_model, aug)
        for i, segm in enumerate(segms):
            segm = segm * 255
            segm = segm.astype(np.uint8)
            Image.fromarray(segm).save(f"{cfg.keyframe_path}/segm_{i}.png")
        return segms
    
    def init_attention_processors(
        self, pipeline, self_attn_mask, cross_attn_mask, fixed_token_indices, full_attn_share=False
    ):
        attn_procs = {}
        unet = pipeline.unet
        number_of_self, number_of_cross = 0, 0
        for i, name in enumerate(unet.attn_processors.keys()):
            is_self_attention = "attn1" in name
            if is_self_attention:
                number_of_self += 1
                attn_procs[name] = SharedSelfAttentionProcessor(self_attn_mask, full_attn_share)
            else:
                number_of_cross += 1
                attn_procs[name] = SharedCrossAttentionProcessor(
                    cross_attn_mask, fixed_token_indices
                )

        unet.set_attn_processor(attn_procs)
    
    def process(self, cfg):
        """
        Process all frames
        no output, save processed keyframes to cfg.keyframes_path/video_name/{original_idx}_keyframe_processed.png
        """
        self_attn_mask = []
        cross_attn_mask = []
        keyframes = get_keyframes(cfg)


        segms = []
        for idx, frame in enumerate(tqdm(keyframes)):
            segm = np.array(Image.open(f"{cfg.keyframe_path}/segm_{idx}.png").convert("L"))
            segm = np.where(segm == 255, 1, 0)
            h, w = segm.shape[-2:]
            segm = np.array(
                Image.fromarray(segm.astype(np.uint8)).resize(
                    (w // 8, h // 8), resample=Image.NEAREST
                )
            )
            segms.append(torch.tensor(segm.flatten()))
        
        segms = torch.stack(segms).to(cfg.device)
        segms = segms.expand(2, -1)
        self_attn_mask = self.get_self_attn_mask(segms)
        cross_attn_mask = self.get_cross_attn_mask(segms)

        print(f"self_attn_mask shape: {self_attn_mask.shape}, size in memory: {self_attn_mask.element_size() * self_attn_mask.nelement() / 1024 / 1024:.2f} MB")
        print(f"cross_attn_mask shape: {cross_attn_mask.shape}, size in memory: {cross_attn_mask.element_size() * cross_attn_mask.nelement() / 1024 / 1024:.2f} MB")

        
        original_promp_embeds, edit_prompt_embeds, fixed_token_indices = (
            self.process_prompts(cfg)
        )

        latents = self.get_inverse_latents(keyframes, original_promp_embeds, self.pipeline.vae)

        init_shared_norm(self.pipeline, full_attn_share=False)
        self.init_attention_processors(
            self.pipeline, self_attn_mask, cross_attn_mask, fixed_token_indices, full_attn_share=False
        )
        processed_keyframes = self.pipeline(
            latents=latents,
            prompt_embeds=edit_prompt_embeds,
        ).images

        save_processed_keyframes(processed_keyframes, cfg)
