import hydra

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # OmegaConf....
    print(OmegaConf.to_yaml(cfg))
    # 对 config 操作
    CLEVAM_DM(cfg)

if __name__ == "__main__":
    main()