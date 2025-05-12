import torch

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
            