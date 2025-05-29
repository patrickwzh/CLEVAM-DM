import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from src.utils import extract_keyframes


def main():
    print("Reading config file...")
    cfg = OmegaConf.load("src/config/config.yaml")
    print("Config file loaded. Extracting keyframes...")
    assert (cfg.prompts.original_inside == cfg.prompts.edit_inside) ^ (cfg.prompts.original_outside == cfg.prompts.edit_outside), "You must only change one of the original or edit prompts."
    if cfg.prompts.original_inside == cfg.prompts.edit_inside:
        cfg.change_background = True
    else:
        cfg.change_background = False
    extract_keyframes(cfg)
    # Write cfg back after modification
    OmegaConf.save(cfg, "src/config/config.yaml")
    print("Keyframes extracted, config determined.")


if __name__ == "__main__":
    main()
