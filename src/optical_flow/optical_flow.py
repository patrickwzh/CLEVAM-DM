import cv2
import torch
import numpy as np
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils
import src.utils as utils
from src.model.utils import get_segmentation_masks
from tqdm import tqdm
import os
import src.optical_flow.PFlowVFI_V0 as pf
def get_optical_flows(model, imgs, chunk_size):
    """
    Get optical flow between iamges.
    imgs: a list of cv2.imread() images
    """
    model.eval()
    io_adapter = IOAdapter(model, imgs[0].shape[:2])
    inputs = io_adapter.prepare_inputs(imgs) # (1, N, 3, H, W)
    # we want to put this in (t-1, 2, c, h, w), where the 2 is every pair of frames
    input_images = inputs["images"][0]
    with torch.no_grad():
        video1 = input_images[:-1]
        video2 = input_images[1:]
        input_images = torch.stack((video2, video1), dim=1).to(model.device)
        inputs["images"] = input_images

        num_pairs = input_images.shape[0]
        all_flows = []

        for start in tqdm(range(0, num_pairs, chunk_size), desc="Forward optical flow"):
            end = min(start + chunk_size, num_pairs)
            chunk_inputs = {k: (v[start:end] if isinstance(v, torch.Tensor) and v.shape[0] == num_pairs else v) for k, v in inputs.items()}
            # print(chunk_inputs['images'].shape)
            predictions = model(chunk_inputs)
            flows = predictions['flows']  # (chunk, 1, 2, H, W)
            all_flows.append(flows.cpu())

        flows_forward = torch.cat(all_flows, dim=0).squeeze(1)  # (N-1, 2, H, W)

        input_images = torch.stack((video1, video2), dim=1).to(model.device)
        inputs["images"] = input_images
        all_flows = []

        for start in tqdm(range(0, num_pairs, chunk_size), desc="Backward optical flow"):
            end = min(start + chunk_size, num_pairs)
            chunk_inputs = {k: (v[start:end] if isinstance(v, torch.Tensor) and v.shape[0] == num_pairs else v) for k, v in inputs.items()}
            predictions = model(chunk_inputs)
            flows = predictions['flows']
            all_flows.append(flows.cpu())
        flows_backward = torch.cat(all_flows, dim=0).squeeze(1)
    return flows_forward.detach().cpu().numpy(), flows_backward.detach().cpu().numpy()

def warp_single_flow(img, flow):
    """
    Warp an image using optical flow.
    img: a cv2.imread() image
    flow: a numpy tensor of shape (2, H, W)
    """
    h, w = img.shape[:2]
    # Create mesh grid for pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    # Compute mapping by adding flow to coordinates
    map_x = (grid_x + flow[0]).astype(np.float32)
    map_y = (grid_y + flow[1]).astype(np.float32)
    next_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return next_img

def interpolate_forward_backward(forwards, backwards):
    assert len(forwards) == len(backwards)
    tot_len = len(forwards)
    forwards = [f.astype(np.float32) for f in forwards]
    backwards = [b.astype(np.float32) for b in backwards]
    interpolates = []
    for i, (forward, backward) in enumerate(zip(forwards, backwards)):
        interpolates.append((((tot_len - i) * forward.astype(np.float16) + i * backward.astype(np.float16)) / tot_len).astype(np.uint8))
    return interpolates

def frame_interpolation(cfg):
    per_vfi = pf.Network(dilate_size=7).to(cfg.device)
    per_vfi.eval()
    
    model = ptlflow.get_model(cfg.optical_flow.model_name, ckpt_path=cfg.optical_flow.ckpt_path)
    model = model.to(cfg.device)
    frames = utils.get_frames(cfg)
    # # # # # Convert frames from HWC (BGR) to CHW (RGB)
    # # # # # frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for f in frames]
    print("\tGetting optical flows...")
    forward_flows, backward_flows = get_optical_flows(model, frames, cfg.chunk_size)
    flows_forward_tensor = torch.from_numpy(forward_flows).float().to(cfg.device)
    flows_backward_tensor = torch.from_numpy(backward_flows).float().to(cfg.device)
    # # # # print(f"flow shape: {flows.shape}")
    # # # os.makedirs(cfg.output_dir, exist_ok=True)
    # # np.save(os.path.join(cfg.output_dir, "flows_rev.npy"), flows)
    # np.save(os.path.join(cfg.output_dir, "flows_fwd.npy"), forward_flows)
    # np.save(os.path.join(cfg.output_dir, "flows_bwd.npy"), backward_flows)
    # forward_flows = np.load(os.path.join(cfg.output_dir, "flows_fwd.npy"))
    # backward_flows = np.load(os.path.join(cfg.output_dir, "flows_bwd.npy"))
    # print(f"shapes: {forward_flows.shape}, {backward_flows.shape}")
    keyframes = utils.get_processed_keyframes(cfg)
    print(len(keyframes), "keyframes found")
    # keyframes = utils.get_keyframes(cfg)
    # # keyframes = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for f in keyframes]
    print("Optical flows obtained. Getting segmentation masks for every frame...")
    masks = get_segmentation_masks(cfg, frames, save_segm=False)
    # np.save(os.path.join(cfg.output_dir, "masks_rev.npy"), masks)
    # masks = np.load(os.path.join(cfg.output_dir, "masks.npy"))
    # masks = np.expand_dims(masks, axis=-1)
    # print(f"masks shape: {masks.shape}")

    processed_frames = []
    next_keyframes = keyframes[1:] + [frames[-1]]
    tot_idx_len = len(keyframes)
    tot_frames = len(frames)
    print("Segmentation finished. Warping frames...")
    print(f"shapes: {frames[0].shape}, {masks[0].shape}, {keyframes[0].shape}")
    # for idx, (keyframe, next_keyframe) in tqdm(enumerate(zip(keyframes, next_keyframes)), desc="Processing keyframes"):
    #     frame_idx = idx * cfg.interval
    #     keyframe = keyframe * masks[frame_idx] + frames[frame_idx] * (1 - masks[frame_idx])
    #     keyframe = keyframe.astype(np.uint8)
    #     processed_frames.append(keyframe)
    #     if idx == tot_idx_len - 1:
    #         warped_frame_forward = keyframe
    #         for j in range(tot_frames - idx * cfg.interval - 1):
    #             flow_idx = idx * cfg.interval + j
    #             flow = forward_flows[flow_idx]
    #             warped_frame_forward = warp_single_flow(warped_frame_forward, flow)
    #             warped_frame_forward = warped_frame_forward * masks[flow_idx] + frames[flow_idx] * (1 - masks[flow_idx])
    #             processed_frames.append(warped_frame_forward.astype(np.uint8))
    #     else:
    #         warped_frames_forward, warped_frames_backward = [], []
    #         warped_frame_forward, warped_frame_backward = keyframe, next_keyframe
    #         for j in range(cfg.interval):
    #             forward_flow_idx = idx * cfg.interval + j
    #             backward_flow_idx = (idx + 1) * cfg.interval - j - 1
    #             warped_frame_forward = warp_single_flow(warped_frame_forward, forward_flows[forward_flow_idx])
    #             warped_frame_forward = warped_frame_forward * masks[forward_flow_idx] + frames[forward_flow_idx] * (1 - masks[forward_flow_idx])
    #             warped_frame_backward = warp_single_flow(warped_frame_backward, backward_flows[backward_flow_idx])
    #             warped_frame_backward = warped_frame_backward * masks[backward_flow_idx] + frames[backward_flow_idx] * (1 - masks[backward_flow_idx])
    #             warped_frames_forward.append(warped_frame_forward)
    #             warped_frames_backward.append(warped_frame_backward)
    #         warped_frames_forward.pop()
    #         warped_frames_backward.pop()
    #         warped_frames_backward.reverse()
    #         processed_frames.extend(interpolate_forward_backward(warped_frames_forward, warped_frames_backward))
    for idx in tqdm(range(len(keyframes)-1), desc="PerVFI Interpolation"):
        kf0 = torch.from_numpy(keyframes[idx]).permute(2,0,1).float().to(cfg.device)/255.0
        kf1 = torch.from_numpy(keyframes[idx+1]).permute(2,0,1).float().to(cfg.device)/255.0
        
        start_idx = idx * cfg.interval
        end_idx = (idx+1) * cfg.interval
        fflow_seg = flows_forward_tensor[start_idx:end_idx]
        bflow_seg = flows_backward_tensor[start_idx:end_idx]
        
        for j in range(cfg.interval):
            t = (j + 1) / cfg.interval  
            
            with torch.no_grad():
                inputs = [kf0.unsqueeze(0), kf1.unsqueeze(0), 
                         fflow_seg[j].unsqueeze(0), bflow_seg[j].unsqueeze(0)]
                
                pred, _ = per_vfi.decode(z_list=None, inps=inputs, time=t)
                pred = (pred.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                
                mask = masks[start_idx + j]
                blended = pred * mask + frames[start_idx + j] * (1 - mask)
                processed_frames.append(blended.astype(np.uint8))
    
    # np.savez_compressed(os.path.join(cfg.output_dir, "processed_frames.npz"), frames=np.array(processed_frames))
    # processed_frames = np.load(os.path.join(cfg.output_dir, "processed_frames.npz"))['frames']
    # processed_frames = processed_frames.astype(np.uint8)
    # processed_frames = np.load(os.path.join(cfg.output_dir, "processed_frames.npz"))['frames']
    print("Warping finished. Saving processed video...")
    utils.save_processed_video(processed_frames, cfg)
