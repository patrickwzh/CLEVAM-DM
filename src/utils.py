import torch
import cv2
import os
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
            keyframe_path = os.path.join(keyframe_path, f"keyframe_{keyframe_id:04d}.png")
            cv2.imwrite(keyframe_path, frame)
            keyframe_paths.append(keyframe_path)
            keyframe_indices.append(frame_count)
            keyframe_id += 1

        frame_count += 1

    cap.release()
    # return keyframe_paths
    return


def get_keyframes(cfg):
    """
    Load keyframes from cfg.output_dir/keyframe_path/keyframe_{keyframe_id}.png
    Return a torch tensor of shape (num_frames, channels, height, width)
    """
    keyframe_path = cfg.keyframe_path
    keyframe_paths = []
    for root, dirs, files in os.walk(keyframe_path):
        for file in files:
            if file.endswith(".png"):
                keyframe_paths.append(os.path.join(root, file))

    keyframes = []
    for path in keyframe_paths:
        frame = cv2.imread(path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # HWC to CHW
        keyframes.append(frame)

    return torch.stack(keyframes)


def save_processed_keyframes(processed_keyframes, cfg):
    """
    Save processed keyframes to cfg.output_dir/{original_idx}_keyframe_processed.png
    """
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(processed_keyframes):
        frame = frame.permute(1, 2, 0).numpy()  # CHW to HWC
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, f"{i:04d}_keyframe_processed.png")
        cv2.imwrite(output_path, frame)
