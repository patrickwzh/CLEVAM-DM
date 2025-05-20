import hydra
from omegaconf import OmegaConf
from src.utils import extract_keyframes, get_keyframes
from src.model.consistent_local_edit import ConsistentLocalEdit

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # OmegaConf....
    # print(OmegaConf.to_yaml(cfg))
    extract_keyframes(cfg)
    # 对 config 操作
    # CLEVAM_DM(cfg)
    model = ConsistentLocalEdit(cfg)
    # model.prepare_segms(cfg, get_keyframes(cfg))
    model.process(cfg)
    

if __name__ == "__main__":
    main()