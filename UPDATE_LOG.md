## 2025.05.29

by lkh

- Update gradio UI for the pipeline
- Update `README.md` with running instructions

## 2025.05.29

by wzh

- Utilize PerVFI for frame interpolation
- Restructure input format, see `README.md` for details
- Update `requirements.txt` with PerVFI dependency


## 2025.05.28

by wzh

- Add CLE support for inversion of original image
- Add support for only changing background


## 2025.05.23

by wzh

- Optimize memory usage for pipeline
- Debug overall workflow
- Update `README.md` with running instructions
- TODOs
  - Update instructions for downloading pretrained weights in `README.md`
  - Clear up the repo
  - Better optical flow pipeline?


## 2025.05.22

by wzh

- Implement optical flow image warping for video reconstruction
- Merge `lang-sam` with optical flow pipeline
- Write running script for overall workflow (prevent OOM)


## 2025.05.22

by lkh

- Use `lang-sam` for segmentation
- Update `requirements.txt`

## 2025.05.22

by lkh

- Rebuild pipeline with BrushNet
- Before running the pipeline, download `segmentation_mask_brushnet_ckpt` and `realisticVisionV60B1_v51VAE` from https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n and fill `brushnet_path` in `config.yaml`.
- Comment out a line of code that calls `_init_weights` if error occurs.
- **TODO**
  - replace maskformer with a better segmentation model.
  - Update `Requirements.txt`, `numpy >= 2.0` can also work now.

## 2025.05.21

by lkh, wzh

- Pipeline work for multiple images
- The image color is too light, may be due to some scale problems

## 2025.05.21

by lkh, wzh

- Use `copy background` to improve the quality of the output image
- The pipeline can work well for single image now!
- **Problem**:
  - The VRAM usage is too high, need to find the problem and optimize.

## 2025.05.10

by lkh

- Change shepe guided diffusion only to the first layer of unet
- The pipeline work for single image now
- **Problems**:
  - `get_inversion_latents` function does not work for batch size > 1
  - The output cannot reserve the shape

## 2025.05.19

by lhz

* update the utils and storation structure
* Move the frame_interpolation to CLEVAM-DM
* Now only support two stages, TODO: change the interface

```
--- {vedio_path}
    --- example.mp4
    --- keyframes({keyframe_path}})
	--- keyframe_0001.png
	--- keyframe_0002.png
	...
--- {OUTPUT_DIR}
    --- output_video.mp4
```

## 2025.05.19

by wzh

- Update requirements and refactor segmentation and attention processing
- Integrated `infer_maskformer` for segmentation in `consistent_local_edit.py`.
- Introduced `maskformer.py` for MaskFormer model setup and inference.
- Removed unused `shape_guided_diffusion` files to clean up the codebase.

## 2025.05.19

by lkh

- Finish `SharedCrossAttentionProcessor` and some helper functions in `src/model/utils.py`.
- Implement `process_prompts` function in `src/model/consistent_local_edit.py`.
- Implement `get_clip` function in `src/model/utils.py`.

## 2025.05.19

by wzh

- Finish `SharedSelfAttentionProcessor` and some helper functions in `src/model/utils.py`.
- Implement some functions in `src/model/consistent_local_edit.py`: `init_attention_processors`, `get_inverse_latents`, `process`.
- **Further TODOs** for consistent local edit:
  - Helper functions in `src/utils.py`
  - Segmentation part in `src/model/consistent_local_edit.py`
  - Helper functions in `src/model/consistent_local_edit.py` and `get_clip` function in `src/model/utils.py`

## 2025.05.18

by wzh

TODOs for consistent local edit:

- Use `StableDiffusionPipeline` as in StyleAligned
- Run DDIM inversion process for the original image first, then use that latent as the input for the pipeline (remember to scale by `1 / pipeline.scheduler.init_noise_sigma`, ref. `pipeline.prepare_latents`).
- Self attention:
  - Edit based on the original `SharedAttentionProcessor` class, modify scaled dot product attention: after softmax, only keep inside-to-inside \& outside-to-outside parts.
  - Rename it to `SharedSelfAttentionProcessor`.
- Cross attention:
  - Implement a new class for shared cross attention, `SharedCrossAttentionProcessor`.
  - For the text prompts, we already have that the first half of the text prompt is for inside, and the second half is for outside.
  - Similar to above, modify the scaled dot product attention: after softmax, only keep inside-text-to-inside-image \& outside-text-to-outside-image parts.
  - Scale the fixed tokens.

## 2025.05.12

by wzh

- Update meeting note
- Reconstruct repo
- Push tips:
  - Avoid using `git add .`, only pick specific files to stage.
  - Stage config files only if necessary (we don't want our local `video_path` to be overwritten by yours).
  - Ensure code and directory structure is maintained, write `utils.py` if necessary.
  - Write docstrings for interface-level functions and classes.

## 2025.04.27

by lhz

- Pipeline init
- directory Structure

```
--- {INPUT_DIR}
    --- input_vedio.mp4
--- {OUTPUT_DIR}
    --- pictures
        --- keyframes
            --- keyframe_0001.png
            --- keyframe_0002.png
            ...
        --- gapframes
            --- gapframe_0001
                --- gapframe_0001_0001.png
                --- gapframe_0001_0002.png
                ...
            --- gapframe_0002
                --- gapframe_0002_0001.png
                --- gapframe_0002_0002.png
    --- output_video.mp4
```

## 2025.04.27

by wzh

- Inital commit
