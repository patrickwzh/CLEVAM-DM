import hydra
from omegaconf import OmegaConf
from src.utils import extract_keyframes, get_keyframes
from src.model.consistent_local_edit import ConsistentLocalEdit
from src.optical_flow.optical_flow import frame_interpolation

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # OmegaConf....
    # print(OmegaConf.to_yaml(cfg))
    extract_keyframes(cfg)
    # 对 config 操作
    # CLEVAM_DM(cfg)
    model = ConsistentLocalEdit(cfg)
    model.process(cfg)

    frame_interpolation(cfg)
    

if __name__ == "__main__":
    main()