'''
    CODE FOR GENERATING PROMPTS USING GRID POINTS
'''
import os
import json
import argparse
from tqdm import tqdm
import imageio.v2 as iio

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import utils

# CONFIGURATIONS
parser = argparse.ArgumentParser()
# DATASET
parser.add_argument("--dataset", type=str, default="mevis")
parser.add_argument("--data_type", type=str, default="valid_u")
# MODELS
parser.add_argument("--bin_size", type=int, default=8)
parser.add_argument("--sam2_cfg", type=str, default="pretrained_models/sam2_hiera_l.yaml")
parser.add_argument("--sam2_ckpt", type=str, default="pretrained_models/sam2_hiera_large.pt")
# PARALLEL
parser.add_argument("--pid", type=int, default=0, help="Index of the current processing segment")
parser.add_argument("--n_pid", type=int, default=1, help="Total number of segments")
args = parser.parse_args()

# DATA TYPE
data_type_dict = {
    "mevis": ["train", "valid", "valid_u"],
    "ref-ytbvos": ["train", "valid", "test"],
    "ref-davis": ["train", "valid"],
}
assert args.data_type in data_type_dict[args.dataset], f"DATA TYPE MUST BE IN {data_type_dict[args.dataset]}"

# DATA DIR
data_dir_dict = {
    "mevis": "datasets/mevis",
    "ref-ytbvos": "datasets/ref-ytbvos",
    "ref-davis": "datasets/ref-davis",
}
data_dir = os.path.join(data_dir_dict[args.dataset], args.data_type, "JPEGImages")

# PROMPT MASK DIR
prompt_masks_dir = os.path.join("sam2_prompts/grid_prompts", args.dataset, args.data_type)
os.makedirs(prompt_masks_dir, exist_ok=True)

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("DEVICE: CUDA")
else:
    device = torch.device("cpu")
    print("DEVICE: CPU")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# LOAD SAM2
sam2 = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=device, apply_postprocess=False)
sam2_amg = SAM2AutomaticMaskGenerator(sam2)

# PROCESS
videos = os.listdir(data_dir)
videos.sort()
videos = videos[args.pid::args.n_pid]
for idx, video_id in enumerate(tqdm(videos, ascii="░▒▓█", colour="CYAN")):
    if os.path.exists(os.path.join(prompt_masks_dir, f"{video_id}.json")):
        continue

    # LOAD FRAMES
    frames = os.listdir(os.path.join(data_dir, video_id))
    frames = sorted(frames)
    if args.bin_size > 0:
        bin_size = args.bin_size
    else:
        n_frames = len(frames)
        bin_size = n_frames // 2
    frames = frames[::bin_size]

    # PROMPT MASK INFOS
    prompt_mask_infos = {
        "video_id": video_id,
        "bin_size": bin_size,
        "prompt_masks": [],
    }
    for bin_idx, frame in enumerate(frames):
        # GET FRAME
        frame_idx = bin_idx * bin_size
        frame = iio.imread(os.path.join(data_dir, video_id, frame))
        frame_area = frame.shape[0] * frame.shape[1]
        
        # GENERATE MASKS
        sam2_mask_infos = sam2_amg.generate(frame)

        if len(sam2_mask_infos) == 0:
            continue

        # FILTER PART MASKS
        sam2_mask_infos = sorted(sam2_mask_infos, key=lambda x: x['area'], reverse=True)
        sam2_masks = torch.stack([torch.from_numpy(sam2_mask_info['segmentation']).float() for sam2_mask_info in sam2_mask_infos]).to(device)
        n_sam2_masks = len(sam2_mask_infos)
        is_part = torch.tensor([False] * n_sam2_masks)
        for sam2_mask_idx in range(n_sam2_masks - 1):
            if is_part[sam2_mask_idx]:
                continue
            full_mask = sam2_masks[sam2_mask_idx]
            P = utils.compute_P(sam2_masks, full_mask)
            is_part[P > 0.7] = True
            is_part[sam2_mask_idx] = False
        
        # SAVE PROMPT MASK INFOS
        for sam2_mask_info, is_part_ in zip(sam2_mask_infos, is_part):
            if is_part_:
                continue
            prompt_mask_infos["prompt_masks"].append({
                "segmentation": utils.encode_rle_mask(sam2_mask_info['segmentation']),
                "stability_score": sam2_mask_info['stability_score'],
                "area": sam2_mask_info['area'],
                "area_ratio": sam2_mask_info['area'] / frame_area,
                "frame_idx": frame_idx,
            })
    
    # SORT PROMPT MASKS
    prompt_mask_infos["prompt_masks"] = sorted(prompt_mask_infos["prompt_masks"], key=lambda x: x["area"], reverse=True)
    for prompt_id, prompt_mask_info in enumerate(prompt_mask_infos["prompt_masks"]):
        prompt_mask_infos["prompt_masks"][prompt_id]["prompt_id"] = prompt_id
    
    # SAVE PROMPT MASK INFOS
    with open(os.path.join(prompt_masks_dir, f"{video_id}.json"), "w") as f:
        json.dump(prompt_mask_infos, f, indent=4)
