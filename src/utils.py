import torch
import cv2
import os
import numpy as np
def extract_keyframes(cfg):
    """
    Load video
    input: cfg.video_path/video_name.mp4
    no outputs, save keyframes to cfg.keyframes_path/video_name/key_frame_{id}.png
    """
    video_path = cfg.video_path
    keyframe_path = cfg.keyframe_path
    interval = cfg.interval  
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keyframe_id = 0
    keyframe_indices = [] 
    keyframe_paths = []    

    if not os.path.exists(keyframe_path):
        os.makedirs(keyframe_path)
    print(f"Extracting keyframes from {video_path} to {keyframe_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            keyframe_path_id = os.path.join(keyframe_path, f"keyframe_{keyframe_id:04d}.png")
            cv2.imwrite(keyframe_path_id, frame)
            keyframe_paths.append(keyframe_path_id)
            keyframe_indices.append(frame_count)
            keyframe_id += 1

        frame_count += 1

    cap.release()
    # return keyframe_paths
    return


def get_keyframes(cfg):
    """
    Load keyframes from cfg.output_dir/keyframe_path/keyframe_{keyframe_id}.png
    Returns a torch tensor of shape (num_frames, height, width, channels) with BGR format
    """
    keyframe_path = cfg.keyframe_path
    keyframe_paths = []
    for root, dirs, files in os.walk(keyframe_path):
        for file in files:
            if file.endswith(".png") and file.startswith("keyframe_"): # TODO: change this because processed keyframes is also like this
                keyframe_paths.append(os.path.join(root, file))

    keyframes = []
    for path in keyframe_paths:
        frame = cv2.imread(path)
        frame = torch.from_numpy(frame)
        keyframes.append(frame)

    return torch.stack(keyframes)


def save_processed_keyframes(processed_keyframes, cfg):
    """
    Processed_keyframes: Image objects
    Save processed keyframes to cfg.output_dir/{original_idx}_keyframe_processed.png
    """
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(processed_keyframes):
        frame = np.array(frame)  # Convert PIL Image to numpy array (HWC, RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(output_path, frame)
