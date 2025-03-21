'''
    CODE FOR GENERATING PROMPTS USING GT MASKS (MEVIS)
'''
import os
import json
import utils
import torch
import argparse
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
import warnings
warnings.filterwarnings(action='ignore')

# CONFIGURATIONS
parser = argparse.ArgumentParser()
# DATASET
parser.add_argument('--dataset', type=str, default='mevis', help='dataset name')
parser.add_argument('--data_type', type=str, default='train', help='train / valid_u / valid')
# MODELS
parser.add_argument("--sam2_cfg", type=str, default="pretrained_models/sam2_hiera_l.yaml")
parser.add_argument("--sam2_ckpt", type=str, default="pretrained_models/sam2_hiera_large.pt")
# ETC
parser.add_argument('--save_prec_rec_iou', action='store_true')
parser.add_argument("--pid", type=int, default=0, help="Index of the current processing segment")
parser.add_argument("--n_pid", type=int, default=1, help="Total number of segments")
args = parser.parse_args()

# DATA TYPE
data_type_dict = {
    "mevis": ["train", "valid_u", "valid"],
}
assert args.data_type in data_type_dict[args.dataset], f"DATA TYPE MUST BE IN {data_type_dict[args.dataset]}"

# DATA DIR
data_dir_dict = {
    "mevis": "datasets/mevis"
}
data_dir = os.path.join(data_dir_dict[args.dataset], args.data_type)
# SAM2 OUTPUT DIR
sam2_output_dir = os.path.join("sam2_tracks/gt_tracks", args.dataset, args.data_type)
os.makedirs(sam2_output_dir, exist_ok=True)

# LOAD META DATA
with open(os.path.join(data_dir_dict[args.dataset], args.data_type, "meta_expressions.json"), "r") as f:
    meta = json.load(f)

# LOAD GT MASKS
if args.save_prec_rec_iou:
    assert args.data_type in ["train", "valid_u"], f"DATA TYPE MUST BE 'train' OR 'valid_u' TO SAVE PRECISION, RECALL, IOU"
    with open(os.path.join(data_dir_dict[args.dataset], args.data_type, "mask_dict.json"), "r") as f:
        mask_dict = json.load(f)

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device("cpu")
print(f"DEVICE: {device}")

# LOAD SAM2
video_predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device)

# PROCESS
runtime_info = {}
for video_idx, (video_id, video_meta) in enumerate(meta['videos'].items()):
    # PARALLEL
    if args.n_pids > 1 and video_idx % args.n_pids != args.pid:
        continue

    print(f"PROCESSING [ {video_idx} / {len(meta['videos'])} ] - {video_id}")
    os.makedirs(os.path.join(sam2_output_dir, "sam2_masklets", video_id), exist_ok=True)
    os.makedirs(os.path.join(sam2_output_dir, "sam2_object_tokens", video_id), exist_ok=True)
    
    # LOAD FRAMES
    frames = os.listdir(os.path.join(data_dir, "JPEGImages", video_id))
    frames.sort()
    n_frames = len(frames)

    # GT MASKLETS
    if args.save_prec_rec_iou:
        gt_masklets = {}
        for expression_id, expression_meta in video_meta['expressions'].items():
            for anno_id in expression_meta['anno_id']:
                anno_id = str(anno_id)
                if anno_id not in gt_masklets:
                    gt_masklets[anno_id] = utils.decode_rle_masklet(mask_dict[anno_id])
    
    # INIT SAM2 VIDEO PREDICTOR
    inference_state = video_predictor.init_state(video_path=os.path.join(data_dir, "JPEGImages", video_id))

    anno_id = 0
    for gt_anno_id in gt_masklets:
        # GET PROMPT MASKS
        prompt_mask_infos = utils.get_prompt_masks(gt_masklets[gt_anno_id])
        
        # GET TOKENS
        for prompt_idx, prompt_mask_info in enumerate(prompt_mask_infos):
            # PROMPT MASK INFOS
            prompt_frame_idx = prompt_mask_info['frame_idx']
            prompt_mask = prompt_mask_info['mask']

            pred_masklet = [None] * n_frames
            pred_tokens = [None] * n_frames

            # PROMPT MASK
            video_predictor.reset_state(inference_state)
            out_frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=prompt_frame_idx,
                obj_id=0,
                mask=prompt_mask,
            )
            pred_masklet[out_frame_idx] = (out_mask_logits > 0.0).float()

            # PROPAGATION
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                pred_masklet[out_frame_idx] = (out_mask_logits > 0.0).float()
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=True):
                pred_masklet[out_frame_idx] = (out_mask_logits > 0.0).float()
            pred_masklet = torch.stack(pred_masklet, dim=0).squeeze()
            assert pred_masklet.ndim == 3, f"pred_masklet.ndim MUST BE 3, BUT {pred_masklet.ndim}"

            # GET TOKENS
            for output_type in ['cond', 'non_cond']:
                for frame_idx, output_dict in inference_state['output_dict_per_obj'][0][f'{output_type}_frame_outputs'].items():
                    pred_tokens[frame_idx] = output_dict['obj_ptr']
            pred_tokens = torch.cat(pred_tokens, dim=0).cpu().numpy()
            assert pred_tokens.ndim == 2, f"pred_tokens.ndim MUST BE 2, BUT {pred_tokens.ndim}"

            pred_masklet_infos = {
                "anno_id": anno_id,
                "rle": utils.encode_rle_masklet_torch(pred_masklet),
                "prompt_type": "GT MASK",
            }

            # COMPUTE PRECISION, RECALL, IOU
            if args.save_prec_rec_iou:
                pred_masklet_infos["precision"] = {}
                pred_masklet_infos["recall"] = {}
                pred_masklet_infos["iou"] = {}
                for gt_anno_id in gt_masklets:
                    gt_masklet = torch.from_numpy(gt_masklets[gt_anno_id]).float().to(device)
                    precision, recall, iou = utils.compute_mask_metrics(
                        pred_masks=pred_masklet,
                        gt_masks=gt_masklet,
                        reduction="mean",
                    )
                    pred_masklet_infos["precision"][gt_anno_id] = precision.item()
                    pred_masklet_infos["recall"][gt_anno_id] = recall.item()
                    pred_masklet_infos["iou"][gt_anno_id] = iou.item()
            
            # SAVE
            with open(os.path.join(sam2_output_dir, "sam2_masklets", video_id, f"{anno_id:05d}.json"), "w") as f:
                json.dump(pred_masklet_infos, f, indent=4)
            np.save(os.path.join(sam2_output_dir, "sam2_object_tokens", video_id, f"{anno_id:05d}.npy"), pred_tokens)
            anno_id += 1