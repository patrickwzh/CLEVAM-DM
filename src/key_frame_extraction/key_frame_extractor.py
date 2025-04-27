import cv2
import os

def extract_keyframes(video_path, output_dir, interval=10):
    """
    Extract keyframes from a video at fixed intervals.
    Args:
        video_path: vedio path
        output_dir: key frame save directory
        interval: interval between adjacent keyframes
    Returns:
        keyframe_paths: a vector containing keyframes
        frame_gaps: a vector containing the number of frames between adjacent keyframes
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keyframe_id = 0
    keyframe_indices = []  # 记录关键帧的原始帧序号
    keyframe_paths = []    # 关键帧保存路径
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            keyframe_path = os.path.join(output_dir, f"keyframe_{keyframe_id:04d}.png")
            cv2.imwrite(keyframe_path, frame)
            keyframe_paths.append(keyframe_path)
            keyframe_indices.append(frame_count)
            keyframe_id += 1
        
        frame_count += 1
    
    cap.release()
    
    # 计算相邻关键帧之间的帧数间隔
    frame_gaps = [
        keyframe_indices[i+1] - keyframe_indices[i] 
        for i in range(len(keyframe_indices)-1)
    ]
    
    return keyframe_paths, frame_gaps
