'''
    CODE FOR GENERATING PROMPTS USING GROUNDING DINO
'''
import os
import cv2
import json
import argparse
from PIL import Image
import imageio.v2 as iio
from tqdm import tqdm

import torch
import numpy as np

from prompt_generator import PromptGenerator
from utils import encode_rle_mask, decode_rle_mask, compute_mask_iou_torch, put_text_box

import warnings
warnings.filterwarnings('ignore')

# CONFIGURATIONS
parser = argparse.ArgumentParser()
# DATASET
parser.add_argument("--dataset", type=str, default="mevis")
parser.add_argument("--data_type", type=str, default="valid_u")
# PROMPT GENERATION
parser.add_argument("--bin_size", type=int, default=4)
parser.add_argument("--box_threshold", type=float, default=0.2)
parser.add_argument("--text_threshold", type=float, default=0.25)
# MODELS
parser.add_argument("--sam2_cfg", type=str, default="pretrained_models/sam2_hiera_l.yaml")
parser.add_argument("--sam2_ckpt", type=str, default="pretrained_models/sam2_hiera_large.pt")
parser.add_argument("--gdino_cfg", type=str, default="pretrained_models/GroundingDINO_SwinT_OGC.py")
parser.add_argument("--gdino_ckpt", type=str, default="pretrained_models/groundingdino_swint_ogc.pth")
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
print(f"DATASET: {args.dataset}, DATA TYPE: {args.data_type}")

# DATA DIR
data_dir_dict = {
    "mevis": "datasets/mevis",
    "ref-ytbvos": "datasets/ref-ytbvos",
    "ref-davis": "datasets/ref-davis",
}
data_dir = os.path.join(data_dir_dict[args.dataset], args.data_type, "JPEGImages")
print(f"DATA DIR: {data_dir}")

# PROMPT MASK DIR
prompt_masks_dir = os.path.join("sam2_prompts/gdino_prompts", args.dataset, args.data_type)
print(f"PROMPT MASK DIR: {prompt_masks_dir}")
os.makedirs(prompt_masks_dir, exist_ok=True)

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("DEVICE: CUDA")
else:
    device = torch.device("cpu")
    print("DEVICE: CPU")

# LOAD PROMPT GENERATOR
print("LOADING PROMPT GENERATOR")
prompt_generator = PromptGenerator(
    # grounding dino
    grounding_dino_cfg=args.gdino_cfg,
    grounding_dino_ckpt=args.gdino_ckpt,
    # sam2
    sam2_cfg=args.sam2_cfg,
    sam2_ckpt=args.sam2_ckpt,
    device=device,
    # etc
    box_threshold=args.box_threshold,
    text_threshold=args.text_threshold,
)

# LOAD META
print("LOADING META")
if args.dataset in ['mevis']:
    with open(os.path.join(data_dir_dict[args.dataset], args.data_type, "meta_expressions.json"), "r") as f:
        meta = json.load(f)
elif args.dataset in ['ref-ytbvos', 'ref-davis']:
    with open(os.path.join(data_dir_dict[args.dataset], "meta_expressions", args.data_type, "meta_expressions.json"), "r") as f:
        meta = json.load(f)
else:
    raise ValueError(f"DATASET MUST BE IN {data_dir_dict.keys()}")

# LOAD GT MASKLETS
print("LOADING GT MASKLETS")
if args.dataset in ['mevis']:
    if args.data_type in ["train", "valid_u"]:
        with open(os.path.join(data_dir_dict[args.dataset], args.data_type, "mask_dict.json"), "r") as f:
            gt_masklet_dict = json.load(f)
    elif args.data_type in ["valid"]:
        gt_masklet_dict = None
elif args.dataset in ['ref-ytbvos', 'ref-davis']:
    gt_masklet_dict = None
else:
    raise ValueError(f"DATASET MUST BE IN {data_dir_dict.keys()}")

# GENERATE PROMPTS
print("GENERATING PROMPTS")
for video_idx, (video_id, video_meta) in enumerate(meta['videos'].items()):
    # PARALLEL
    if args.n_pid > 1 and video_idx % args.n_pid != args.pid:
        continue
    if os.path.exists(os.path.join(prompt_masks_dir, f"{video_id}.json")):
        continue

    # LOAD FRAMES
    frames = os.listdir(os.path.join(data_dir, video_id))
    frames.sort()
    frames = frames[::args.bin_size]

    # INITIALIZATION
    gt_mask_ious_per_expression = {}
    prompt_masks_per_video = {
        "video_id": video_id,
        "bin_size": args.bin_size,
        "prompt_masks": [],
    }
    expressions = []
    for expression_idx, (expression_id, expression_meta) in enumerate(video_meta['expressions'].items()):
        assert str(expression_idx) == expression_id, f"expression_idx {expression_idx} is not equal to expression_id {expression_id}"
        expressions.append(expression_meta['exp'])
        if gt_masklet_dict is not None:
            gt_anno_ids = expression_meta['anno_id']
            gt_mask_ious_per_expression[expression_id] = {
                str(gt_anno_id): 0.0 for gt_anno_id in gt_anno_ids
            }
    
    # GENERATE PROMPTS PER FRAME
    for bin_idx, frame_path in enumerate(tqdm(frames, desc=f"VIDEO {video_id} [{video_idx} / {len(meta['videos'])}]", ascii="░▒▓█", colour="CYAN")):
        # LOAD FRAME
        frame_idx = bin_idx * args.bin_size
        frame = Image.open(os.path.join(data_dir, video_id, frame_path)).convert("RGB")
        
        # GENERATE PROMPTS
        prompt_masks = prompt_generator.generate_prompts(raw_image=frame, raw_texts=expressions)

        # COMPUTE IOU
        for expression_id in video_meta['expressions']:
            # GT MASKS
            expression = video_meta['expressions'][expression_id]['exp']
            if gt_masklet_dict is not None:
                gt_anno_ids = video_meta['expressions'][expression_id]['anno_id']
                # get gt masks
                gt_masks = {}
                for gt_anno_id in gt_anno_ids:
                    gt_mask = gt_masklet_dict[str(gt_anno_id)][frame_idx]
                    if gt_mask is None:
                        gt_masks[str(gt_anno_id)] = None
                    else:
                        gt_masks[str(gt_anno_id)] = torch.from_numpy(decode_rle_mask(gt_mask)).float().to(device)
            # PROMPT MASKS
            tokenized = prompt_masks[expression_id]['tokenized']
            for prompt_mask_info in prompt_masks[expression_id]['preds']:
                if gt_masklet_dict is not None:
                    pred_mask = torch.from_numpy(prompt_mask_info['sam2_mask']).float().to(device)
                    iou_dict = {}
                    for gt_anno_id in gt_anno_ids:
                        if gt_masks[str(gt_anno_id)] is None:
                            iou_dict[gt_anno_id] = {"iou": 0.0}
                            continue
                        iou = compute_mask_iou_torch(pred_mask, gt_masks[str(gt_anno_id)])
                        iou_dict[gt_anno_id] = {"iou": iou}
                        gt_mask_ious_per_expression[expression_id][str(gt_anno_id)] = max(gt_mask_ious_per_expression[expression_id][str(gt_anno_id)], iou)
                    prompt_masks_per_video['prompt_masks'].append({
                        "segmentation": encode_rle_mask(pred_mask.cpu().numpy()),
                        "stability_score": prompt_mask_info['stability_score'],
                        "score": prompt_mask_info['mask_score'],
                        "area": pred_mask.sum().item(),
                        "area_ratio": pred_mask.sum().item() / (pred_mask.shape[0] * pred_mask.shape[1]),
                        "frame_idx": frame_idx,
                        "pred_bbox": prompt_mask_info['bbox'].tolist(),
                        "pred_phrase": prompt_mask_info['phrase'],
                        "token_score": prompt_mask_info['token_score'],
                        "expression_id": expression_id,
                        "metrics": iou_dict,
                    })
                else:
                    pred_mask = prompt_mask_info['sam2_mask']
                    prompt_masks_per_video['prompt_masks'].append({
                        "segmentation": encode_rle_mask(pred_mask),
                        "stability_score": prompt_mask_info['stability_score'],
                        "score": prompt_mask_info['mask_score'],
                        "area": int(pred_mask.sum()),
                        "area_ratio": float(pred_mask.sum() / (pred_mask.shape[0] * pred_mask.shape[1])),
                        "frame_idx": frame_idx,
                        "pred_bbox": prompt_mask_info['bbox'].tolist(),
                        "pred_phrase": prompt_mask_info['phrase'],
                        "token_score": prompt_mask_info['token_score'],
                        "expression_id": expression_id,
                        "metrics": {},
                    })
    
    # SORT PROMPT MASKS
    prompt_masks_per_video['prompt_masks'] = sorted(prompt_masks_per_video['prompt_masks'], key=lambda x: x['area'], reverse=True)
    for prompt_id, prompt_mask_info in enumerate(prompt_masks_per_video['prompt_masks']):
        prompt_mask_info['prompt_id'] = prompt_id
    
    # SAVE PROMPT
    with open(os.path.join(prompt_masks_dir, f"{video_id}.json"), "w") as f:
        json.dump(prompt_masks_per_video, f, indent=4)
