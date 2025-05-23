import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from src.utils import extract_keyframes, get_keyframes
from src.model.consistent_local_edit import ConsistentLocalEdit
from src.optical_flow.optical_flow import frame_interpolation


def main():
    print("Reading config file...")
    cfg = OmegaConf.load("src/config/config.yaml")
    print("Config file loaded. Extracting keyframes...")
    extract_keyframes(cfg)
    print("Keyframes extracted.")


if __name__ == "__main__":
    main()
