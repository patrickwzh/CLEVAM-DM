import cv2
import torch
import numpy as np
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
import src.utils as utils
from src.model.utils import get_segmentation_masks
from tqdm import tqdm
from src.frame_interpolation.models.generators import build_generator_arch
from src.frame_interpolation.models.pipeline import interpolate_single_frame
import os


def get_optical_flows(model, imgs, cfg):
    """
    Get optical flow between iamges.
    imgs: a list of cv2.imread() images
    """
    model.eval()
    io_adapter = IOAdapter(model, imgs[0].shape[:2])
    inputs = io_adapter.prepare_inputs(imgs)  # (1, N, 3, H, W)
    input_images = inputs["images"][0]
    
    with torch.no_grad():
        total_frames = len(input_images)
        
        # Calculate which frames are keyframes
        keyframe_indices = list(range(0, total_frames, cfg.interval))
        
        # Initialize flows lists with zeros for all frames
        zero_flow = torch.zeros((2, input_images.shape[2], input_images.shape[3]), device=model.device)
        flows_forward_list = [zero_flow.clone() for _ in range(total_frames)]
        flows_backward_list = [zero_flow.clone() for _ in range(total_frames)]
        
        # Forward flows - from current keyframe to each frame until next keyframe
        for ki, keyframe_idx in enumerate(keyframe_indices[:-1]):
            next_keyframe_idx = keyframe_indices[ki + 1]
            keyframe = input_images[keyframe_idx]
            
            for frame_idx in range(keyframe_idx + 1, next_keyframe_idx + 1):
                frame = input_images[frame_idx]
                
                # Stack keyframe and current frame for flow calculation
                pair = torch.stack([keyframe, frame], dim=0).unsqueeze(0)
                
                # Prepare inputs for the model
                chunk_inputs = {
                    k: (v[0:1] if isinstance(v, torch.Tensor) and len(v.shape) > 0 else v)
                    for k, v in inputs.items()
                }
                chunk_inputs["images"] = pair.to(model.device)
                
                # Calculate optical flow
                predictions = model(chunk_inputs)
                flow = predictions["flows"].squeeze(0).squeeze(0)  # (2, H, W)
                
                # Store the flow for this frame
                flows_forward_list[frame_idx] = flow
        
        # Backward flows - from next keyframe to each frame after previous keyframe
        for ki, keyframe_idx in enumerate(keyframe_indices[1:], 1):
            prev_keyframe_idx = keyframe_indices[ki - 1]
            keyframe = input_images[keyframe_idx]
            
            for frame_idx in range(prev_keyframe_idx, keyframe_idx):
                frame = input_images[frame_idx]
                
                # Stack keyframe and current frame for flow calculation (reversed order)
                pair = torch.stack([keyframe, frame], dim=0).unsqueeze(0)
                
                # Prepare inputs for the model
                chunk_inputs = {
                    k: (v[0:1] if isinstance(v, torch.Tensor) and len(v.shape) > 0 else v)
                    for k, v in inputs.items()
                }
                chunk_inputs["images"] = pair.to(model.device)
                
                # Calculate optical flow
                predictions = model(chunk_inputs)
                flow = predictions["flows"].squeeze(0).squeeze(0)  # (2, H, W)
                
                # Store the flow for this frame
                flows_backward_list[frame_idx] = flow
        
        # Convert to tensors of shape (total_frames, 2, H, W)
        flows_forward = torch.stack(flows_forward_list, dim=0)
        flows_backward = torch.stack(flows_backward_list, dim=0)
    return flows_forward.detach().cpu().numpy(), flows_backward.detach().cpu().numpy()


def warp_single_flow(img, flow):
    """
    Warp an image using optical flow.
    Utility for checking the correctness of optical flow.
    img: a cv2.imread() image
    flow: a numpy tensor of shape (2, H, W)
    """
    h, w = img.shape[:2]
    # Create mesh grid for pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    # Compute mapping by adding flow to coordinates
    map_x = (grid_x + flow[0]).astype(np.float32)
    map_y = (grid_y + flow[1]).astype(np.float32)
    next_img = cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    return next_img


def frame_interpolation(cfg):
    model = ptlflow.get_model(
        cfg.optical_flow.model_name, ckpt_path=cfg.optical_flow.ckpt_path
    )
    model = model.to(cfg.device)
    frames = utils.get_frames(cfg)

    print("\tGetting optical flows...")
    forward_flows, backward_flows = get_optical_flows(model, frames, cfg)

    keyframes = utils.get_processed_keyframes(cfg)
    print(len(keyframes), "keyframes found")

    print("Optical flows obtained. Getting segmentation masks for every frame...")
    masks = get_segmentation_masks(cfg, frames, save_segm=False)
    if cfg.change_background:
        masks = 1 - masks

    processed_frames = []
    tot_idx_len = len(keyframes)
    print("Segmentation finished. Preparing for frame interpolation...")

    # Load PerVFI model
    generator, model_file = "v00", cfg.pervfi_path
    net = build_generator_arch(generator)
    state_dict = {
        k.replace("module.", ""): v for k, v in torch.load(model_file).items()
    }
    net.load_state_dict(state_dict)
    net = net.to(cfg.device)
    net.eval()

    # Process each keyframe pair
    for idx in tqdm(range(tot_idx_len - 1), desc="Processing keyframe pairs"):
        current_keyframe_idx = idx * cfg.interval
        next_keyframe_idx = (idx + 1) * cfg.interval

        # Get the processed keyframes
        img0 = keyframes[idx] * masks[current_keyframe_idx] + frames[
            current_keyframe_idx
        ] * (1 - masks[current_keyframe_idx])
        img1 = keyframes[idx + 1] * masks[next_keyframe_idx] + frames[
            next_keyframe_idx
        ] * (1 - masks[next_keyframe_idx])

        # Convert to uint8 and add the first keyframe to processed frames
        img0 = img0.astype(np.uint8)
        img1 = img1.astype(np.uint8)

        if idx == 0:
            processed_frames.append(img0)

        # Direct flows between keyframes (fflow and bflow)
        fflow = forward_flows[next_keyframe_idx]
        bflow = backward_flows[current_keyframe_idx]

        # Process frames between keyframes
        for middle_idx in range(current_keyframe_idx + 1, next_keyframe_idx):
            # Calculate time parameter (0 to 1)
            time = (middle_idx - current_keyframe_idx) / cfg.interval

            # Calculate accumulated flows from keyframes to middle frame
            F1t = forward_flows[middle_idx]
            F2t = backward_flows[middle_idx]

            # Convert numpy arrays to PyTorch tensors
            img0_tensor = (
                torch.from_numpy(img0)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(cfg.device)
            )
            img1_tensor = (
                torch.from_numpy(img1)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(cfg.device)
            )
            fflow_tensor = (
                torch.from_numpy(fflow)
                .float()
                .unsqueeze(0)
                .to(cfg.device)
            )
            bflow_tensor = (
                torch.from_numpy(bflow)
                .float()
                .unsqueeze(0)
                .to(cfg.device)
            )
            F1t_tensor = (
                torch.from_numpy(F1t)
                .float()
                .unsqueeze(0)
                .to(cfg.device)
            )
            F2t_tensor = (
                torch.from_numpy(F2t)
                .float()
                .unsqueeze(0)
                .to(cfg.device)
            )

            # Perform interpolation
            with torch.no_grad():
                interpolated_frame = interpolate_single_frame(
                    net,
                    img0_tensor / 255.0,
                    img1_tensor / 255.0,
                    fflow_tensor,
                    bflow_tensor,
                    F1t_tensor,
                    F2t_tensor,
                    time,
                ) * 255.0

            # Convert back to numpy and apply mask
            interpolated_np = (
                interpolated_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            )
            mask_idx = middle_idx
            interpolated_np = interpolated_np * masks[mask_idx] + frames[mask_idx] * (
                1 - masks[mask_idx]
            )
            if cfg.save_mid_interpolated_frames and middle_idx % cfg.save_interval == cfg.save_interval // 2:
                cv2.imwrite(
                    os.path.join(cfg.output_dir, f"interpolated_frame_{middle_idx:04d}.png"),
                    interpolated_np.astype(np.uint8),
                )
            processed_frames.append(interpolated_np.astype(np.uint8))

        # Add the next keyframe after interpolating all intermediate frames
        if idx == tot_idx_len - 2:
            processed_frames.append(img1)

    print("Frame interpolation finished. Saving processed video...")
    utils.save_processed_video(processed_frames, cfg)
