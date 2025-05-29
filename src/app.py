import gradio as gr
from omegaconf import OmegaConf
import os
import torch
import shutil
from src.utils import extract_keyframes, get_keyframes
from src.model.consistent_local_edit import ConsistentLocalEdit
from src.frame_interpolation.optical_flow import frame_interpolation
from src.model.utils import get_segmentation_masks
import gc

def process_video(input_video, original_inside, edit_inside, original_outside, edit_outside, guidance_scale, time_per_keyframe, chunk_size):
    yield gr.update(visible=True), "Reading config file..."
    cfg = OmegaConf.load("src/config/config.yaml")
    cfg.prompts.original_inside = original_inside
    cfg.prompts.original_outside = original_outside
    cfg.prompts.edit_inside = edit_inside
    cfg.prompts.edit_outside = edit_outside
    cfg.guidance_scale = guidance_scale
    cfg.video_path = input_video
    cfg.time_per_keyframe = time_per_keyframe
    cfg.chunk_size = chunk_size

    log = []

    # clear
    os.system(f"rm {cfg.keyframe_path}* {cfg.frame_path}* {cfg.output_dir}*")
    log.append("Intermediate files cleared. Extracting keyframes...")
    yield gr.update(visible=True), show_log(log)

    # extract keyframes
    extract_keyframes(cfg)
    log.append("Keyframes extracted. Segmenting keyframes...")
    yield gr.update(visible=True), show_log(log)

    # segm
    keyframes = get_keyframes(cfg)
    get_segmentation_masks(cfg, keyframes, save_segm=True)
    log.append("Keyframes segmented. Beginning consistent local edit...")
    yield gr.update(visible=True), show_log(log)

    # cle
    model = ConsistentLocalEdit(cfg)
    model.process(cfg)
    log.append("Consistent local edit completed. Beginning frame interpolation via optical flow warping...")
    yield gr.update(visible=True), show_log(log)

    del model
    gc.collect()
    torch.cuda.empty_cache()


    # interpolation
    frame_interpolation(cfg)
    output_video = os.path.join(cfg.output_dir, "processed_video.mp4")
    log.append("Frame interpolation completed. Enjoy!")
    yield output_video, show_log(log)

def show_log(log):
    log_text = "\n\n".join(log)
    return gr.Markdown(f"Processing log:\n\n{log_text}")


with gr.Blocks(title="CLEVAM-DM") as app:
    gr.Markdown("# CLEVAM-DM")

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video", include_audio=False)

            original_inside = gr.Textbox(label="Original Inside Prompt", placeholder="Describe the original inside scene")
            edit_inside = gr.Textbox(label="Edit Inside Prompt", placeholder="Describe the edited inside scene")
            original_outside = gr.Textbox(label="Original Outside Prompt", placeholder="Describe the original outside scene")
            edit_outside = gr.Textbox(label="Edit Outside Prompt", placeholder="Describe the edited outside scene")

            guidance_scale = gr.Slider(minimum=1, maximum=12, value=7.5, step=0.1, label="Guidance Scale")
            time_per_keyframe = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Keyframe Sample Interval (seconds)")
            chunk_size = gr.Number(value=20, label="Chunk Size")

        with gr.Column():
            output_video = gr.Video(label="Output Video")
            log_text = gr.Markdown("Processing log:")
    submit_btn = gr.Button("Process Video")
    submit_btn.click(
        process_video,
        inputs=[input_video, original_inside, edit_inside, original_outside, edit_outside, guidance_scale, time_per_keyframe, chunk_size],
        outputs=[output_video, log_text],
        show_progress="full"
    )

app.launch()    