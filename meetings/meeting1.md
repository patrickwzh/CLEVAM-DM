# Meeting 1

给一个视频以及 text prompt (把 A 换成 B), 然后输出成品视频.

先给原视频取出关键帧, 然后对每个关键帧进行 consistent local edit, 然后通过 optical flow (ref. DLSS) 进行插帧, 最后合成视频. 注意可能需要用原视频的图片来计算 motion, 因为生成的视频之间可能没发完全对齐.

重点在于 consistent local edit.

目前的工作:

1. Shape-guided diffusion
   做的 "local edit" 部分. 具体操作是把 diffusion u-net 中的 latent space 的 attention map 进行 mask. 论文中的做法是直接用 segmentation model 去 segment 出物体, 然后把这个原图大小的 mask 直接 resize (PIL.Image.NEAREST) 到 latent space 的大小 (512 // 8) (ref. sgdiff, LDM latent space is a perceptually equivalent downsampled version of image space, meaning we are able to apply Inside-Outside Attention in latent space via downsampled object masks). 主体部分是 inside attention, 外边部分是 outside attention.

    具体做法是, 对于 text-image cross-attention, 把每个 text embedding 按照指代 inside/outside 进行区分, 然后对于 inside 相关的 text embedding cross-attn, 通过 mask 只允许他修改主体部分的内容, vice versa. 对于 image self-attn, 也分开处理.

2. Style-aligned
   生成一系列 consistent 的图片. 也是对 u-net 里的 attention map 进行操作. 具体做法是, 有一个 reference image, 然后对每个后续生成的图片, attention 操作时把 reference 的 key \& value 也 concat 到 新图片 attention 中. 然后用到了 AdaIN, 是在 shared attention 之前, 把新图片的 query, key 的 mean 和 std 对齐到 reference 的 mean 和 std. 这样就可以生成 consistent 的图片了.

综合两个 work:

1. 提取关键帧, 对每个关键帧:
2. 通过 text prompt segment 出主体和背景
3. 对第一张关键帧, 进行 shape-guided diffusion, 生成 reference
4. 对于接下来的关键帧, shared-attn 只对 self-attn 部分生效. 所以对 cross-attn:
    - text prompt 关于主体的部分, 把 image 的 attn map 进行 mask, 只允许修改主体部分
    - text prompt 关于背景的部分, 把 image 的 attn map 进行 mask, 只允许修改背景部分
      对 self-attn:
    - inside (主体) 部分, 把当前图片的 attn map mask 成主体的形状, 然后和 reference image 的 attn map (也要只有主体的部分) 进行 shared attn
    - outside (背景) 部分, 把当前图片的 attn map mask 成背景的形状, 然后和 reference image 的 attn map (也要只有背景的部分) 进行 shared attn
      **问题:** 具体的 attn 过程是什么样的? mask 是 0, 1 吗? 还是有 -infty? inside/outside attn 是在哪里交流的 (否则会完全不 consistent)?
5. 对于每个关键帧, 进行 optical flow, 然后插帧

hydra config

```yaml
video_path: /path/to/video.mp4
video_name: video_name
output_path: /path/to/output
keyframes_path: /path/to/keyframes
```

```python
def extract_keyframes(cfg):
    """
    Load video
    input: cfg.video_path/video_name.mp4
    no outputs, save keyframes to cfg.keyframes_path/video_name/{original_idx}_keyframe.png
    """
    pass
```

```python
class ConsistentLocalEdit:
    def process_frame(self, frame, cfg):
        """
        处理单个关键帧, 对第一帧和后面不一样
        第一帧简单, 后面复杂,
        看代码
        no output, save processed keyframe to cfg.keyframes_path/video_name/{original_idx}_keyframe_processed.png
        """
        pass
    def process(self, cfg):
        """
        总处理, 需要 cfg.
        """
        for frame in keyframes:
            self.process_frame(frame, cfg)
```

```python
class OpticalFlow:
    def __init__(self, cfg):
        self.cfg = cfg
    def process(self, cfg):
        """
        处理关键帧, 插帧
        save output video to cfg.output_path/video_name.mp4
        """
        pass
```

```python
@hydra.main(config_path="config.yaml")
def main(cfg):
    # OmegaConf....
    print(OmegaConf.to_yaml(cfg))
    # 对 config 操作
    CLEVAM(cfg)

if __name__ == "__main__":
    main(cfg)
```

lhz 先做 `extract_keyframes` 和 `OpticalFlow`; wzh 和 lkh 看看 `CLEVAM`
