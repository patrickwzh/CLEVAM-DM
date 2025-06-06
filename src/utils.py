import torch
import cv2
import os
import numpy as np

def extract_keyframes(cfg):
    """
    Load video
    input: cfg.video_path/video_name.mp4
    no outputs, save keyframes to cfg.keyframes_path/video_name/key_frame_{id}.png
    Also sets cfg.fps to the video's frames per second.
    """
    video_path = cfg.video_path
    keyframe_path = cfg.keyframe_path
    frame_path = cfg.frame_path
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keyframe_id = 0
    keyframe_indices = [] 
    keyframe_paths = []    

    # Read fps and write to cfg
    fps = cap.get(cv2.CAP_PROP_FPS)
    cfg.fps = fps
    cfg.interval = int(fps * cfg.time_per_keyframe)
    interval = cfg.interval

    if not os.path.exists(keyframe_path):
        os.makedirs(keyframe_path)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    print(f"Extracting frames & keyframes from {video_path} to {frame_path} & {keyframe_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to nearest multiple of 8 (smaller than original)
        h, w = frame.shape[:2]

        # Resize so that the max side length <= cfg.max_size, keeping aspect ratio
        max_size = cfg.max_size
        scale = min(1.0, max_size / max(h, w))
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Make sure new_h and new_w are multiples of 8
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        if new_h != h or new_w != w:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if frame_count % interval == 0:
            keyframe_path_id = os.path.join(keyframe_path, f"keyframe_{keyframe_id:04d}.png")
            cv2.imwrite(keyframe_path_id, frame)
            keyframe_paths.append(keyframe_path_id)
            keyframe_indices.append(frame_count)
            keyframe_id += 1
        
        frame_path_id = os.path.join(frame_path, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path_id, frame)

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
        keyframes.append(frame)

    return keyframes

def get_frames(cfg):
    """
    Load frames from cfg.output_dir/frame_path/frame_{frame_id}.png
    Returns a list of numpy arrays (num_frames, height, width, channels) with BGR format
    """
    frame_path = cfg.frame_path
    frame_paths = []
    for root, dirs, files in os.walk(frame_path):
        for file in files:
            if file.endswith(".png") and file.startswith("frame_"):
                frame_paths.append(os.path.join(root, file))

    # Sort frame_paths by frame index extracted from filename
    def extract_index(path):
        filename = os.path.basename(path)
        # Assumes format: frame_0000.png
        idx_str = filename.replace("frame_", "").replace(".png", "")
        return int(idx_str)
    frame_paths.sort(key=extract_index)

    frames = []
    for path in frame_paths:
        frame = cv2.imread(path)
        frames.append(frame)

    return frames

def get_processed_keyframes(cfg):
    """
    Load processed keyframes from cfg.output_dir/{idx}_keyframe_processed.png
    Returns a numpy list of shape (num_frames, height, width, channels) with BGR format
    """
    keyframe_path = cfg.output_dir
    keyframe_paths = []
    for root, dirs, files in os.walk(keyframe_path):
        for file in files:
            if file.endswith(".png"):
                keyframe_paths.append(os.path.join(root, file))

    # Sort keyframe_paths by index extracted from filename (e.g., 0000.png)
    def extract_index(path):
        filename = os.path.basename(path)
        idx_str = filename.replace(".png", "")
        return int(idx_str)
    keyframe_paths.sort(key=extract_index)

    keyframes = []
    for path in keyframe_paths:
        frame = cv2.imread(path)
        keyframes.append(frame)

    return keyframes

def save_processed_keyframes(processed_keyframes, cfg, RGB2BGR=True):
    """
    Processed_keyframes: Image objects
    Save processed keyframes to cfg.output_dir/{original_idx}_keyframe_processed.png
    """
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(processed_keyframes):
        frame = np.array(frame)  # Convert PIL Image to numpy array (HWC, RGB)
        if RGB2BGR:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(output_path, frame)

def save_processed_video(frames, cfg):
    """
    Save processed video to cfg.output_dir/processed_video.mp4
    frames: list of numpy arrays (HWC, BGR)
    """
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "processed_video.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(output_path, fourcc, cfg.fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        output_movie.write(frame)
    output_movie.release()
    print(f"Processed video saved to {output_path}")
