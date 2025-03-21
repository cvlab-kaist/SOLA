'''
    CODE FOR GENERATING PROMPTS USING SAM2
'''
import os
import cv2
import json
import time
import utils
import seg_utils
import torch
import argparse
import numpy as np
from tqdm import tqdm
import imageio.v2 as iio
from sam2.build_sam import build_sam2_video_predictor

# CONFIGURATIONS
parser = argparse.ArgumentParser()
# DATASET
parser.add_argument("--dataset", type=str, default="mevis")
parser.add_argument("--data_type", type=str, default="valid_u")
# TOKEN GENERATION
parser.add_argument("--bin_size", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--miou_thresh", type=float, default=0.7)
parser.add_argument("--n_max_tracks", type=int, default=64, help="Maximum number of tracks that can be generated in a video")
# MODELS
parser.add_argument("--sam2_cfg", type=str, default="pretrained_models/sam2_hiera_l.yaml")
parser.add_argument("--sam2_ckpt", type=str, default="pretrained_models/sam2_hiera_large.pt")
# ETC
parser.add_argument("--save_prec_rec_iou", action="store_true")
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
data_dir = os.path.join(data_dir_dict[args.dataset], args.data_type)
# PROMPT MASKS DIR
prompt_masks_dir = os.path.join("sam2_prompts/grid_prompts", args.dataset, args.data_type)
# SAM2 OUTPUT DIR
sam2_output_dir = os.path.join("sam2_tracks/grid_tracks", args.dataset, args.data_type)

# META DATA
if args.dataset in ["mevis"]:
    with open(os.path.join(data_dir, "meta_expressions.json"), "r") as f:
        meta = json.load(f)
elif args.dataset in ["ref-ytbvos", "ref-davis"]:
    with open(os.path.join(data_dir_dict[args.dataset], "meta_expressions", args.data_type, "meta_expressions.json"), "r") as f:
        meta = json.load(f)
else:
    raise ValueError(f"DATASET MUST BE EITHER mevis OR ref-ytbvos OR ref-davis")

# GT MASKLETS
if args.save_prec_rec_iou:
    if args.dataset in ["mevis"]:
        assert args.data_type in ["valid_u", "train"], f"DATA TYPE MUST BE EITHER valid_u OR train"
        with open(os.path.join(data_dir, "mask_dict.json"), "r") as f:
            mask_dict = json.load(f)
        print(f"N ANNO: {len(mask_dict)}")
    elif args.dataset in ["ref-ytbvos", "ref-davis"]:
        assert args.data_type in ["train"], f"DATA TYPE MUST BE train"
    else:
        raise ValueError(f"DATASET MUST BE EITHER mevis OR ref-ytbvos OR ref-davis")

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# SAM2 VIDEO PREDICTOR
video_predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device)

# PROCESSING
runtime_info = {}
for video_idx, (video_id, _) in enumerate(meta['videos'].items()):
    if video_idx % args.n_pids != args.pid:
        continue

    start_time = time.time()
    os.makedirs(os.path.join(sam2_output_dir, "sam2_masklets", video_id), exist_ok=True)
    os.makedirs(os.path.join(sam2_output_dir, "sam2_object_tokens", video_id), exist_ok=True)

    # LOAD FRAMES
    frames = os.listdir(os.path.join(data_dir, "JPEGImages", video_id))
    frames = sorted(frames)

    # GT MASKLETS
    if args.save_prec_rec_iou:
        if args.dataset in ["mevis"]:
            gt_masklets = seg_utils.get_masklets(video_id, meta, mask_dict)
            gt_masklets = {
                gt_anno_id: seg_utils.reshape_masklet(gt_masklets[gt_anno_id]) for gt_anno_id in gt_masklets
            }
        elif args.dataset in ["ref-ytbvos", "ref-davis"]:
            gt_masklets = seg_utils.get_masklets_ytbvos(
                masklet_dir=os.path.join(data_dir, "Annotations", video_id),
                reshape=True,
            )
        else:
            raise ValueError(f"DATASET MUST BE EITHER mevis OR ref-ytbvos OR ref-davis")

    # LOAD PROMPT MASKS
    with open(os.path.join(prompt_masks_dir, f"{video_id}.json"), "r") as f:
        prompt_mask_infos = json.load(f)
    
    assert video_id == prompt_mask_infos['video_id'], f"VIDEO ID MISMATCH: {video_id} != {prompt_mask_infos['video_id']}"
    assert args.bin_size % prompt_mask_infos['bin_size'] == 0, f"BIN SIZE MISMATCH: {args.bin_size} != {prompt_mask_infos['bin_size']}"
    
    prompt_mask_infos = prompt_mask_infos['prompt_masks']
    n_frames = len(frames)
    n_prompt_mask_infos = len(prompt_mask_infos)

    n_not_used = 0
    for prompt_mask_info in prompt_mask_infos:
        prompt_mask_info['segmentation'] = seg_utils.decode_rle_mask(prompt_mask_info['segmentation'])
        prompt_mask_info['status'] = 0 # 0: not tracked, 1: tracked, 2: filtered, 3: not used
        if prompt_mask_info['frame_idx'] % args.bin_size != 0:
            prompt_mask_info['status'] = 3
            n_not_used += 1

    # INIT SAM2 VIDEO PREDICTOR
    inference_state = video_predictor.init_state(video_path=os.path.join(data_dir, "JPEGImages", video_id))

    n_tracked = 0
    n_filtered = 0
    n_iter = 0
    prompt_mask_info_idx = 0
    while True:
        if prompt_mask_info_idx >= n_prompt_mask_infos or n_tracked >= args.n_max_tracks:
            break
        batched_prompt_mask_infos = {
            'frame_idx': None,
            'prompt_infos': [],
        }
        n_tracked_ = 0
        n_filtered_ = 0
        for prompt_mask_info in prompt_mask_infos:
            if prompt_mask_info['status'] == 1:
                n_tracked_ += 1
            elif prompt_mask_info['status'] == 2:
                n_filtered_ += 1
        assert n_tracked_ == n_tracked, f"TRACKED MISMATCH: {n_tracked_} != {n_tracked}"
        assert n_filtered_ == n_filtered, f"FILTERED MISMATCH: {n_filtered_} != {n_filtered}"
        # batched prompt mask infos
        for prompt_mask_info in prompt_mask_infos:
            # already tracked or filtered prompt masks
            if prompt_mask_info['status'] > 0:
                continue
            # init batched prompt mask infos
            if batched_prompt_mask_infos['frame_idx'] is None:
                batched_prompt_mask_infos['frame_idx'] = prompt_mask_info['frame_idx']
                batched_prompt_mask_infos['prompt_infos'].append(prompt_mask_info)
                prompt_mask_info['status'] = 1
            else: # batched prompt mask infos
                if prompt_mask_info['frame_idx'] == batched_prompt_mask_infos['frame_idx']:
                    batched_prompt_mask_infos['prompt_infos'].append(prompt_mask_info)
                    prompt_mask_info['status'] = 1
                else:
                    continue
            # check if the number of tracked masklets is less than the maximum number of masklets
            if n_frames > 200: # large video
                if len(batched_prompt_mask_infos['prompt_infos']) >= 2 or n_tracked + len(batched_prompt_mask_infos['prompt_infos']) >= args.n_max_tracks:
                    break
            else: # small video
                if len(batched_prompt_mask_infos['prompt_infos']) >= args.batch_size or n_tracked + len(batched_prompt_mask_infos['prompt_infos']) >= args.n_max_tracks:
                    break
        if batched_prompt_mask_infos['frame_idx'] is None:
            # no more prompt mask infos to process
            break
        # start sam2 processing
        print(f"VIDEO {video_id} [{video_idx + 1:03d} / {len(meta['videos'])}] | FRAME {int(batched_prompt_mask_infos['frame_idx']):3d}")
        print(f"TOTAL [{n_not_used + n_tracked + n_filtered} / {n_prompt_mask_infos}] | NOT USED [{n_not_used} / {n_prompt_mask_infos}] | TRACKED [{n_tracked} / {n_prompt_mask_infos}] | FILTERED [{n_filtered} / {n_prompt_mask_infos}]")
        print(f"ITER {n_iter + 1:03d} | PROCESSING BATCHED PROMPT MASKS -> {[info['prompt_id'] for info in batched_prompt_mask_infos['prompt_infos']]}")
        n_tracked += len(batched_prompt_mask_infos['prompt_infos'])
        n_iter += 1
        # init state
        video_predictor.reset_state(inference_state)
        # get prompt mask
        pred_output_dict = {
            'prompt_ids': [],
            'masklets': {},
            'tokens': {},
        }
        for prompt_mask_info in batched_prompt_mask_infos['prompt_infos']:
            prompt_id = prompt_mask_info['prompt_id']
            pred_output_dict['prompt_ids'].append(prompt_id)
            prompt_mask = prompt_mask_info['segmentation']
            pred_output_dict['masklets'][prompt_id] = [None] * n_frames
            out_frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=int(batched_prompt_mask_infos['frame_idx']),
                obj_id=prompt_id,
                mask=prompt_mask,
            )
            pred_output_dict['masklets'][prompt_id][out_frame_idx] = (out_mask_logits > 0.0).float()
        # propagate
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            for prompt_idx, prompt_id in enumerate(pred_output_dict['prompt_ids']):
                pred_output_dict['masklets'][prompt_id][out_frame_idx] = (out_mask_logits[prompt_idx] > 0.0).float()
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=True):
            for prompt_idx, prompt_id in enumerate(pred_output_dict['prompt_ids']):
                pred_output_dict['masklets'][prompt_id][out_frame_idx] = (out_mask_logits[prompt_idx] > 0.0).float()
        pred_output_dict['masklets'] = {
            prompt_id: torch.cat(pred_output_dict['masklets'][prompt_id], dim=0) for prompt_id in pred_output_dict['prompt_ids']
        }
        # get pred tokens
        pred_output_dict['tokens'] = {
            prompt_id: [None] * n_frames for prompt_id in pred_output_dict['prompt_ids']
        }
        for output_type in ['cond', 'non_cond']:
            for frame_idx, output_dict in inference_state['output_dict'][f'{output_type}_frame_outputs'].items():
                for prompt_idx, prompt_id in enumerate(pred_output_dict['prompt_ids']):
                    obj_ptr = output_dict['obj_ptr'][prompt_idx]
                    pred_output_dict['tokens'][prompt_id][frame_idx] = obj_ptr
        pred_output_dict['tokens'] = {
            prompt_id: torch.stack(pred_output_dict['tokens'][prompt_id], dim=0).cpu().numpy() for prompt_id in pred_output_dict['prompt_ids']
        }
        # save pred masklets and tokens
        pred_masklet_infos = {}
        for prompt_id in pred_output_dict['prompt_ids']:
            # data format
            pred_masklet_infos[prompt_id] = {
                "anno_id": prompt_id,
                "rle": seg_utils.encode_rle_masklet_torch(pred_output_dict['masklets'][prompt_id]),
                "prompt_type": "SAM2 AMG MASK",
            }
        # resize pred masklets
        pred_output_dict['masklets'] = {
            prompt_id: seg_utils.reshape_masklet(pred_output_dict['masklets'][prompt_id]) for prompt_id in pred_output_dict['prompt_ids']
        }
        # compute precision, recall, iou
        for prompt_id in tqdm(pred_output_dict['prompt_ids'], desc=f"SAVE PRED MASKLETS AND TOKENS"):
            if args.save_prec_rec_iou:
                pred_masklet_infos[prompt_id]["precision"] = {}
                pred_masklet_infos[prompt_id]["recall"] = {}
                pred_masklet_infos[prompt_id]["iou"] = {}
                for gt_anno_id, gt_masklet in gt_masklets.items():
                    precision, recall, iou = utils.compute_mask_metrics(
                        pred_masks=pred_output_dict['masklets'][prompt_id],
                        gt_masks=gt_masklet.to(device),
                    )
                    pred_masklet_infos[prompt_id]["precision"][gt_anno_id] = precision.squeeze().item()
                    pred_masklet_infos[prompt_id]["recall"][gt_anno_id] = recall.squeeze().item()
                    pred_masklet_infos[prompt_id]["iou"][gt_anno_id] = iou.squeeze().item()
            # filtering prompt mask
            for prompt_mask_info in prompt_mask_infos:
                if prompt_mask_info['status'] > 0: # already tracked or filtered prompt masks
                    continue
                pred_mask = pred_output_dict['masklets'][prompt_id][prompt_mask_info['frame_idx']].to(device)
                h, w = pred_mask.shape
                prompt_mask = torch.from_numpy(prompt_mask_info['segmentation']).float().to(device)
                prompt_mask = torch.nn.functional.interpolate(prompt_mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze(0).squeeze(0)
                iou = seg_utils.compute_mask_iou(pred_mask, prompt_mask)
                if iou > args.miou_thresh:
                    prompt_mask_info['status'] = 2
                    prompt_mask_info['filtered_by'] = prompt_id
                    prompt_mask_info['filtered_iou'] = iou
                    n_filtered += 1
            # save pred masklets and tokens
            with open(os.path.join(sam2_output_dir, "sam2_masklets", video_id, f"{prompt_id:05d}.json"), "w") as f:
                json.dump(pred_masklet_infos[prompt_id], f, indent=4)
            np.save(os.path.join(sam2_output_dir, "sam2_object_tokens", video_id, f"{prompt_id:05d}.npy"), pred_output_dict['tokens'][prompt_id])
    print("=" * 100)
    print(f"VIDEO {video_id} DONE | ELAPSED TIME: {time.time() - start_time:.2f} SEC")
    print(f"TOTAL [{n_not_used + n_tracked + n_filtered} / {n_prompt_mask_infos}] | NOT USED [{n_not_used} / {n_prompt_mask_infos}] | TRACKED [{n_tracked} / {n_prompt_mask_infos}] | FILTERED [{n_filtered} / {n_prompt_mask_infos}]")
    print("=" * 100)
    not_used_prompt_ids = [prompt_mask_info['prompt_id'] for prompt_mask_info in prompt_mask_infos if prompt_mask_info['status'] == 3]
    tracked_prompt_ids = [prompt_mask_info['prompt_id'] for prompt_mask_info in prompt_mask_infos if prompt_mask_info['status'] == 1]
    filtered_prompt_ids = [prompt_mask_info['prompt_id'] for prompt_mask_info in prompt_mask_infos if prompt_mask_info['status'] == 2]
    not_tracked_prompt_ids = [prompt_mask_info['prompt_id'] for prompt_mask_info in prompt_mask_infos if prompt_mask_info['status'] == 0]
    if len(tracked_prompt_ids) < args.n_max_tracks:
        assert len(not_tracked_prompt_ids) == 0, f"NOT TRACKED PROMPT MASKS ARE FOUND: {not_tracked_prompt_ids}"
    runtime_info[video_id] = {
        'time': time.time() - start_time,
        'n_frames': n_frames,
        'n_tracked': n_tracked,
        'n_filtered': n_filtered,
        'n_not_used': n_not_used,
        'n_total': n_prompt_mask_infos,
        'batch_size': args.batch_size,
        'not_used_prompt_ids': not_used_prompt_ids,
        'tracked_prompt_ids': tracked_prompt_ids,
        'filtered_prompt_ids': filtered_prompt_ids,
        'not_tracked_prompt_ids': not_tracked_prompt_ids,
    }
    with open(os.path.join(sam2_output_dir, f"runtime_info_{args.bin_size}.json"), "w") as f:
        json.dump(runtime_info, f, indent=4)