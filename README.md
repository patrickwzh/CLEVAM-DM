# CLEVAM-DM: Consistent Local Edits in Videos via Attention Manipulation in Diffusion Models

This repository contains the code and data for our 2025 DL course final project.

## Setup

After cloning the repository, create a conda environment and install the required packages:

```bash
conda create -n clevam python=3.11.11
conda activate clevam
pip install -r requirements.txt
```

Note that you need to manually choose the version for `cupy` in `requirements.txt` based on your CUDA version.

## Running the Pipeline

First, download the necessary pretrained weights.

- Download directory `segmentation_mask_brushnet_ckpt` and `realisticVisionV60B1_v51VAE` from [Google Drive](https://drive.google.com/drive/folders/1fqmS1CEOvXCxNWFrsSYd_jHYXxrydh1n) and place them in `cfg.brushnet_path`.
- Download [pre-trained checkpoint](https://huggingface.co/Mulns/PerVFI-v1-0/tree/main/PerVFI) named `v00.pth` and place it in `cfg.pervfi_path`.

All scripts are located in the `scripts` folder.

To run the pipeline, first specify the configurations in `src/config/config.yaml`. Please make sure all **folder** paths ends with `/`.
If you run into an OOM error, you can try to reduce the `chunk_size` in `config.yaml`.

Configurations you **should not** change:
```yaml
do_inverse: true # set to true to make the edits more aligned
interval: ... # automatically determined by time_per_keyframe
change_background: ... # automatically determined by the prompts
fps: ... # automatically determined by the input video
```

Also, make sure to set the environment variable `CUDA_HOME` to your CUDA installation path, e.g.,

```bash
export CUDA_HOME=/usr/local/cuda-11.4 # you might want to ls /usr/local to find the correct path
```

Then, you can run the pipeline with the following command:

```bash
sh scripts/run.sh
```

If you simply want to clear the intermediate files (**including** the final output video), you can run

```bash
sh scripts/clear.sh
```

You can check out `scripts/run.sh` to run python scripts separately.

Run `python -m src.app` to start the web app.
