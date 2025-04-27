import key_frame_extractor.key_frame_extractor as kfe
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple
import torch
'''
    Remember to store the key frames and gap frames in the same directory, the directory structure should be:
--- {INPUT_DIR}
    --- input_vedio.mp4
--- {OUTPUT_DIR}
    --- pictures
        --- keyframes
            --- keyframe_0001.png
            --- keyframe_0002.png
            ...
        --- gapframes
            --- gapframe_0001
                --- gapframe_0001_0001.png
                --- gapframe_0001_0002.png
                ...
            --- gapframe_0002
                --- gapframe_0002_0001.png
                --- gapframe_0002_0002.png
    --- output_video.mp4
'''



class VideoStyleTransfer:
    def __init__(self, input_dir, output_dir, model, fps=30, device: str = "cuda"):
        """
        初始化视频风格转换类
        Args:
            model_name: 使用的扩散模型名称
            device: 计算设备（cuda/cpu）
        """
        self.device = device
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.fps = fps
    def extract_keyframes(self, prompt, interval: int = 10) -> Tuple[List[str], List[int]]:
        """
        Generate pictures from a video at fixed intervals.
        Args:
            video_path: vedio path
            output_dir: picture save directory
            interval: interval between adjacent pictures
        Returns:
            picture_paths: a vector containing pictures
            frame_gaps: a vector containing the number of frames between adjacent pictures
        """
        # Extract keyframes using the key_frame_extractor module
        video_path = os.path.join(self.input_dir, "input_video.mp4")
        output_dir = os.path.join(self.output_dir, "pictures", "keyframes")
        key_frame_paths, frame_gaps = kfe.extract_keyframes(video_path, output_dir, interval)
        """
        TODO: modify the attention structure in codebase to generate style align transferm and return their directory
        """
        key_frame = self.model.generate_style_align(key_frame_paths, prompt)
        # Save the generated keyframes
        for i, key_frame in enumerate(key_frame):
            key_frame_path = os.path.join(output_dir, f"keyframe_{i:04d}.png")
            cv2.imwrite(key_frame_path, key_frame)
        return key_frame, frame_gaps

    # def generate_style_aligned_keyframes(self, keyframe_paths: List[str], prompt: str, reference_image_path: str) -> List[np.ndarray]:
    #     """
    #     生成风格对齐的关键帧
    #     Args:
    #         keyframe_paths: 原始关键帧路径列表
    #         prompt: 目标风格文本提示
    #         reference_image_path: 参考风格图像路径
    #     Returns:
    #         aligned_frames: 对齐后的关键帧列表（numpy数组）
    #     """
    #     aligned_frames = []
    #     reference_image = Image.open(reference_image_path).convert("RGB")
        
    #     for path in tqdm(keyframe_paths, desc="Processing keyframes"):
    #         # 实际应用中应替换为您的模型调用逻辑
    #         input_image = Image.open(path).convert("RGB")
    #         output_image = self.model(
    #             prompt=prompt,
    #             image=input_image,
    #             reference_image=reference_image,
    #         ).images[0]
            
    #         aligned_frames.append(np.array(output_image))
    #     self.aligned_frames = aligned_frames
    #     return aligned_frames

    def generate_gap_frames(self, keyframes: List[np.ndarray], frame_gaps: List[int]) -> List[np.ndarray]:
        """
        Generate pictures from the gaps between keyframes.
        Args:
            key_pictures: a vector containing keyframes
            frame_gaps: a vector containing the number of frames between adjacent keyframes
            output_dir: picture save directory
        Returns:
            gap_pictures: a vector containing gap pictures
        """
        gap_pictures = []
        gap_paths = os.path.join(self.output_dir, "pictures", "gapframes")
        os.makedirs(gap_paths, exist_ok=True)
        for i in range(len(keyframes) - 1):
            gap = frame_gaps[i]
            start_picture = keyframes[i]
            end_picture = keyframes[i + 1]
            frame_path = os.path.join(gap_paths, f"gapframe_{i:04d}")
            os.makedirs(frame_path, exist_ok=True)
            '''
            TODO: generate consistent frames with given frames, you may need to extract the optical flow to detect the motion, store them in the same directory as the keyframes
            '''
            for j in range(gap):
                pass

        return 

    def export_video(self, frames: List[np.ndarray], output_path: str):
        """
        导出最终视频
        Args:
            frames: 所有帧列表
            output_path: 输出视频路径
        """
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        for frame in tqdm(frames, desc="Exporting video"):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        writer.release()

    def process_video(
        self,
        input_video_path: str,
        reference_image_path: str,
        prompt: str,
        output_dir: str,
        interval: int = 10
    ) -> str:
        """
        完整处理流程
        Args:
            input_video_path: 输入视频路径
            reference_image_path: 参考风格图像路径
            prompt: 目标风格文本提示
            output_dir: 输出目录
            interval: 关键帧间隔
        Returns:
            输出视频路径
        """
        # key frame extraction
        keyframe, frame_gaps = self.extract_keyframes(input_video_path, output_dir, interval)
        
        # gap frames generation
        self.generate_gap_frames(keyframe, frame_gaps)
        
        # load all pictures from the directory
        '''
        TODO: load all pictures from the directory, and store them in a list, then generate a video with the given fps
        '''
        
        return
