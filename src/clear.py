import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import os


def main():
    print("Reading config file...")
    cfg = OmegaConf.load("src/config/config.yaml")
    print("Config file loaded. Clearing intermediate files...")
    os.system(f"rm {cfg.keyframe_path}* {cfg.frame_path}* {cfg.output_dir}*")
    print("Intermediate files cleared.")


if __name__ == "__main__":
    main()
