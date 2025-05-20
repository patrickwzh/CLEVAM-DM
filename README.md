# CLEVAM-DM: Consistent Local Edits in Videos via Attention Manipulation in Diffusion Models

This repository contains the code and data for our 2025 DL course final project.

## Setup

After cloning the repository, create a conda environment and install the required packages:

```bash
conda create -n clevam python=3.10
conda activate clevam
pip install -r requirements.txt
python -m pip install --force-reinstall -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
pip install numpy==1.26.4
```

## Running the Pipeline

All scripts are located in the `scripts` folder.
