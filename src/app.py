import gradio as gr
from omegaconf import OmegaConf
import os
import shutil
from src.utils import extract_keyframes, get_keyframes
from src.model.consistent_local_edit import ConsistentLocalEdit
from src.frame_interpolation.optical_flow import frame_interpolation
from src.model.utils import get_segmentation_masks

def process_video(input_video, original_inside, original_outside, edit_inside, edit_outside, guidance_scale):
    yield gr.update(visible=True), "Reading config file..."
    cfg = OmegaConf.load("src/config/config.yaml")
    cfg.prompts.original_inside = original_inside
    cfg.prompts.original_outside = original_outside
    cfg.prompts.edit_inside = edit_inside
    cfg.prompts.edit_outside = edit_outside
    cfg.guidance_scale = guidance_scale
    cfg.video_path = input_video

    # clear
    os.system(f"rm {cfg.keyframe_path}* {cfg.frame_path}* {cfg.output_dir}*")
    yield gr.update(visible=True), "Intermediate files cleared. Extracting keyframes..."

    # extract keyframes
    extract_keyframes(cfg)
    yield gr.update(visible=True), "Keyframes extracted. Segmenting keyframes..."

    # segm
    keyframes = get_keyframes(cfg)
    get_segmentation_masks(cfg, keyframes, save_segm=True)
    yield gr.update(visible=True), "Keyframes segmented. Beginning consistent local edit..."

    # cle
    model = ConsistentLocalEdit(cfg)
    model.process(cfg)
    yield gr.update(visible=True), "Consistent local edit completed. Beginning frame interpolation via optical flow warping..."

    # interpolation
    frame_interpolation(cfg)
    output_video = os.path.join(cfg.output_dir, "processed_video.mp4")
    yield output_video, "Frame interpolation completed. Enjoy!"


with gr.Blocks(title="CLEVAM-DM") as app:
    gr.Markdown("# CLEVAM_DM")

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video", include_audio=False)

            original_inside = gr.Textbox(label="Original Inside Prompt", placeholder="Describe the original inside scene")
            original_outside = gr.Textbox(label="Original Outside Prompt", placeholder="Describe the original outside scene")
            edit_inside = gr.Textbox(label="Edit Inside Prompt", placeholder="Describe the edited inside scene")
            edit_outside = gr.Textbox(label="Edit Outside Prompt", placeholder="Describe the edited outside scene")

            guidance_scale = gr.Slider(minimum=1, maximum=12, value=7.5, step=0.1, label="Guidance Scale")

        with gr.Column():
            output_video = gr.Video(label="Output Video")
            log_text = gr.Markdown("Processing log:")
    submit_btn = gr.Button("Process Video")
    submit_btn.click(
        process_video,
        inputs=[input_video, original_inside, original_outside, edit_inside, edit_outside, guidance_scale],
        outputs=[output_video, log_text],
        show_progress="full"
    )

app.launch()    