import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from src.utils import extract_keyframes, get_keyframes
from src.model.consistent_local_edit import ConsistentLocalEdit
from src.optical_flow.optical_flow import frame_interpolation
from src.model.utils import get_segmentation_masks


def main():
    print("Reading config file...")
    cfg = OmegaConf.load("src/config/config.yaml")
    print("Config file loaded. Segmenting keyframes...")
    keyframes = get_keyframes(cfg)
    get_segmentation_masks(cfg, keyframes, save_segm=True)
    print("Keyframes segmented.")


if __name__ == "__main__":
    main()
