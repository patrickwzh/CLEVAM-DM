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
)
from diffusers import StableDiffusionPipeline, DDIMScheduler
from src.utils import get_keyframes, save_processed_keyframes


class ConsistentLocalEdit:
    def __init__(self, cfg):
        self.segment = ...
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

    def init_attention_processors(
        pipeline, self_attn_mask, cross_attn_mask, fixed_token_indices
    ):
        attn_procs = {}
        unet = pipeline.unet
        number_of_self, number_of_cross = 0, 0
        for i, name in enumerate(unet.attn_processors.keys()):
            is_self_attention = "attn1" in name
            if is_self_attention:
                number_of_self += 1
                attn_procs[name] = SharedSelfAttentionProcessor(self_attn_mask)
            else:
                number_of_cross += 1
                attn_procs[name] = SharedCrossAttentionProcessor(
                    cross_attn_mask, fixed_token_indices
                )

        unet.set_attn_processor(attn_procs)

    def get_inverse_latents(self, keyframes, original_prompt_embeds):
        latents = inversion(
            keyframes, self.pipeline.unet, self.scheduler, original_prompt_embeds
        )
        # pipeline's `prepare_latents` has scaled the latents by `init_noise_sigma`, so we need to pre-scale it back
        return latents / self.pipeline.scheduler.init_noise_sigma

    def process_prompts(self, cfg):
        _, embedding_temp = get_tokens_embeds(self.clip_tokenizer, self.clip, "")
        prompt_collect_indices = torch.tensor([i for i in range(1, 39)])
        prompt_inside_indices = torch.tensor([i for i in range(1, 39)])
        prompt_outside_indices = torch.tensor([i for i in range(39, 77)])

        original_prompt_embeds = embedding_temp.clone()
        original_inside_tokens, original_inside_embeds = get_tokens_embeds(
            self.clip_tokenizer, self.clip, cfg.prompts.original_inside
        )
        original_outside_tokens, original_outside_embeds = get_tokens_embeds(
            self.clip_tokenizer, self.clip, cfg.prompts.original_outside
        )
        original_prompt_embeds[:, prompt_inside_indices, :] = original_inside_embeds[
            :, prompt_collect_indices, :
        ]
        original_prompt_embeds[:, prompt_outside_indices, :] = original_outside_embeds[
            :, prompt_collect_indices, :
        ]

        edit_prompt_embeds = embedding_temp.clone()
        edit_inside_tokens, edit_inside_embeds = get_tokens_embeds(
            self.clip_tokenizer, self.clip, cfg.prompts.edit_inside
        )
        edit_outside_tokens, edit_outside_embeds = get_tokens_embeds(
            self.clip_tokenizer, self.clip, cfg.prompts.edit_outside
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

    def process(self, cfg):
        """
        总处理, 需要 cfg.
        no output, save processed keyframes to cfg.keyframes_path/video_name/{original_idx}_keyframe_processed.png
        """
        self_attn_mask = []
        cross_attn_mask = []
        keyframes = get_keyframes(cfg)
        for frame in keyframes:
            segm = self.segment(frame)
            self_attn_mask.append(self.get_self_attn_mask(segm))
            cross_attn_mask.append(self.get_cross_attn_mask(segm, cfg))

        original_promp_embeds, edit_prompt_embeds, fixed_token_indices = (
            self.process_prompts(cfg)
        )
        init_shared_norm(self.pipeline)
        self.init_attention_processors(
            self.pipeline, self_attn_mask, cross_attn_mask, fixed_token_indices
        )

        latents = self.get_inverse_latents(keyframes, original_promp_embeds)
        processed_keyframes = self.pipeline(
            latents=latents,
            prompt_embeds=edit_prompt_embeds,
        ).images

        save_processed_keyframes(processed_keyframes, cfg)
