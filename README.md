# CLEVAM-DM: Consistent Local Edits in Videos via Attention Manipulation in Diffusion Models

This repository contains the code and data for our 2025 DL course final project.

## Setup

After cloning the repository, create a conda environment and install the required packages:

```bash
conda create -n clevam python=3.10
conda activate clevam
pip install -r requirements.txt
pip install numpy==1.26.4
```

## Running the Pipeline

First, download the necessary pretrained weights. *TODO*

All scripts are located in the `scripts` folder.

To run the pipeline, first specify the configurations in `src/config/config.yaml`. Please make sure all **folder** paths ends with `/`.
If you run into an OOM error, you can try to reduce the `chunk_size` in `config.yaml`.

Then, you can run the pipeline with the following command:

```bash
sh scripts/run.sh
```

Or if you simply want to clear the intermediate files (**including** the final output video), you can run

```bash
sh scripts/clear.sh
```

You can check out `scripts/run.sh` to run python scripts separately.
