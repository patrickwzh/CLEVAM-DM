import torch

class ConsistentLocalEdit:
    def init_attention_processors(pipeline, self_attn_mask, cross_attn_mask, fixed_token_indices):
        attn_procs = {}
        unet = pipeline.unet
        number_of_self, number_of_cross = 0, 0
        num_self_layers = len([name for name in unet.attn_processors.keys() if 'attn1' in name])
        if style_aligned_args is None:
            only_self_vec = _get_switch_vec(num_self_layers, 1)
        else:
            only_self_vec = _get_switch_vec(num_self_layers, style_aligned_args.only_self_level)
        for i, name in enumerate(unet.attn_processors.keys()):
            is_self_attention = 'attn1' in name
            if is_self_attention:
                number_of_self += 1
                if style_aligned_args is None or only_self_vec[i // 2]:
                    attn_procs[name] = DefaultAttentionProcessor()
                else:
                    attn_procs[name] = SharedAttentionProcessor(style_aligned_args)
            else:
                number_of_cross += 1
                attn_procs[name] = DefaultAttentionProcessor()

        unet.set_attn_processor(attn_procs)
    
    def init_shared_norm(self, pipeline):
        pass
    
    def __init__(self, cfg):
        self.segmentation_model = ...
        self.clip = ...
        self.pipeline = ...
        self.pipeline = self.pipeline.to(cfg.device)
    
    def get_inverse_latents(self, keyframes, cfg):
        pass

    def process_prompts(self, cfg):
        pass

    def save_keyframes(self, processed_keyframes, cfg):
        pass

    def process(self, cfg):
        """
        总处理, 需要 cfg.
        no output, save processed keyframes to cfg.keyframes_path/video_name/{original_idx}_keyframe_processed.png
        """
        self_attn_mask = []
        cross_attn_mask = []
        for frame in keyframes:
            segm = self.segmentation_model(frame)
            self_attn_mask.append(self.get_self_attn_mask(segm))
            cross_attn_mask.append(self.get_cross_attn_mask(segm, cfg))
        
        prompt_embeds, fixed_token_indices = self.process_prompts(cfg)
        self.init_shared_norm(self.pipeline)
        self.init_attention_processors(self.pipeline, self_attn_mask, cross_attn_mask, fixed_token_indices)

        latents = self.get_inverse_latents(keyframes, cfg)
        processed_keyframes = self.pipeline(
            latents=latents,
            prompt_embeds=prompt_embeds,
        ).images

        self.save_keyframes(processed_keyframes, cfg)

            