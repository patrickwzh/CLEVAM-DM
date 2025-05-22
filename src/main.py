import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from src.utils import extract_keyframes, get_keyframes
from src.model.consistent_local_edit import ConsistentLocalEdit

def main():
    # OmegaConf....
    # print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.load('src/config/config.yaml')
    extract_keyframes(cfg)
    # 对 config 操作
    # CLEVAM_DM(cfg)
    model = ConsistentLocalEdit(cfg)
    model.process(cfg)
    

if __name__ == "__main__":
    main()