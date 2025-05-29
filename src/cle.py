import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from src.utils import extract_keyframes, get_keyframes
from src.model.consistent_local_edit import ConsistentLocalEdit


def main():
    print("Reading config file...")
    cfg = OmegaConf.load("src/config/config.yaml")
    print("Config loaded. Beginning consistent local edit...")
    model = ConsistentLocalEdit(cfg)
    model.process(cfg)
    print("Consistent local edit completed.")


if __name__ == "__main__":
    main()
