'''
    CODE FOR GENERATING TOKENS USING GROUNDING DINO
'''
import os
import json
import time
import utils
import seg_utils
import torch
import argparse
import numpy as np
from tqdm import tqdm
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
parser.add_argument("--stability_score_thresh", type=float, default=0.85)
parser.add_argument("--n_max_tracks", type=int, default=16, help="Maximum number of tracks that can be generated in a video")
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
# PROMPT MASK DIR
prompt_masks_dir = os.path.join("sam2_prompts/gdino_prompts", args.dataset, args.data_type)
# SAM2 OUTPUT DIR
sam2_output_dir = os.path.join("sam2_tracks/gdino_tracks", args.dataset, args.data_type)

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
        assert args.data_type in ["valid_u", "train"], f"DATA TYPE MUST BE EITHER 'valid_u' OR 'train' TO SAVE PRECISION, RECALL, IOU IN 'mevis' DATASET"
        with open(os.path.join(data_dir, "mask_dict.json"), "r") as f:
            mask_dict = json.load(f)
        print(f"N ANNO: {len(mask_dict)}")
    elif args.dataset in ["ref-davis"]:
        assert args.data_type in ["train", "valid"], f"DATA TYPE MUST BE EITHER 'train' OR 'valid' TO SAVE PRECISION, RECALL, IOU IN 'ref-davis' DATASET"
    elif args.dataset in ["ref-ytbvos"]:
        assert args.data_type in ["train"], f"DATA TYPE MUST BE 'train' TO SAVE PRECISION, RECALL, IOU IN 'ref-ytbvos' OR 'ref-davis' DATASET"
    else:
        raise ValueError(f"DATASET MUST BE EITHER 'mevis' OR 'ref-ytbvos' OR 'ref-davis'")

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

# PROCESS PER VIDEO
runtime_info = {}
for video_idx, (video_id, video_meta) in enumerate(meta['videos'].items()):
    if args.n_pids > 1 and video_idx % args.n_pids != args.pid:
        continue

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
    assert args.bin_size == prompt_mask_infos['bin_size'], f"BIN SIZE MISMATCH: {args.bin_size} != {prompt_mask_infos['bin_size']}"
    
    prompt_mask_infos = prompt_mask_infos['prompt_masks']
    n_frames = len(frames)
    n_prompt_mask_infos = len(prompt_mask_infos)

    # INIT SAM2 VIDEO PREDICTOR
    print(f"INITIALIZING SAM2 VIDEO PREDICTOR FOR VIDEO {video_id}")
    inference_state = video_predictor.init_state(video_path=os.path.join(data_dir, "JPEGImages", video_id))
    
    # PROCESS PER EXPRESSION
    if os.path.exists(os.path.join(sam2_output_dir, "runtime_info.json")):
        with open(os.path.join(sam2_output_dir, "runtime_info.json"), "r") as f:
            runtime_info = json.load(f)
    runtime_info[video_id] = runtime_info.get(video_id, {})

    for expression_id, expression_meta in video_meta['expressions'].items():
        if expression_id in runtime_info[video_id]:
            continue

        runtime_info[video_id][expression_id] = {}
        start_time = time.time()

        os.makedirs(os.path.join(sam2_output_dir, "sam2_masklets", video_id, f"{expression_id}"), exist_ok=True)
        os.makedirs(os.path.join(sam2_output_dir, "sam2_object_tokens", video_id, f"{expression_id}"), exist_ok=True)

        n_tracked, n_filtered, n_not_used, n_total = 0, 0, 0, 0
        
        # FILTER PROMPT MASKS
        prompt_mask_infos_expression = []
        for prompt_mask_info in prompt_mask_infos:
            if prompt_mask_info['expression_id'] == expression_id:
                n_total += 1
                prompt_mask_info['segmentation'] = seg_utils.decode_rle_mask(prompt_mask_info['segmentation'])
                prompt_mask_info['status'] = 0 # 0: not tracked, 1: tracked, 2: filtered, 3: not used
                if prompt_mask_info['frame_idx'] % args.bin_size != 0 or prompt_mask_info['stability_score'] < args.stability_score_thresh:
                    prompt_mask_info['status'] = 3
                    n_not_used += 1
                else:
                    prompt_mask_infos_expression.append(prompt_mask_info)
        
        # TRACK PROMPT MASKS
        while True:
            if n_tracked >= args.n_max_tracks:
                break

            # BATCH PROMPT MASKS
            batched_prompt_mask_infos = {
                'frame_idx': None,
                'prompt_infos': [],
            }
            for prompt_mask_info in prompt_mask_infos_expression:
                if prompt_mask_info['status'] > 0:
                    # already tracked or filtered
                    continue
                if batched_prompt_mask_infos['frame_idx'] is None:
                    # first prompt mask
                    batched_prompt_mask_infos['frame_idx'] = prompt_mask_info['frame_idx']
                    batched_prompt_mask_infos['prompt_infos'].append(prompt_mask_info)
                    prompt_mask_info['status'] = 1
                    n_tracked += 1
                else:
                    if prompt_mask_info['frame_idx'] == batched_prompt_mask_infos['frame_idx']:
                        # same frame
                        batched_prompt_mask_infos['prompt_infos'].append(prompt_mask_info)
                        prompt_mask_info['status'] = 1
                        n_tracked += 1
                    else:
                        # different frame
                        break
                if n_frames > 200 and len(batched_prompt_mask_infos['prompt_infos']) >= 2:
                    break
                if len(batched_prompt_mask_infos['prompt_infos']) >= args.batch_size:
                    break
                if len(batched_prompt_mask_infos['prompt_infos']) + n_tracked >= args.n_max_tracks:
                    break
            
            if batched_prompt_mask_infos['frame_idx'] is None:
                # no more prompt masks to track
                break

            # START SAM2 PROCESSING
            print(f"VIDEO {video_id} [ {video_idx} / {len(meta['videos'])} ] | EXP {expression_id} | FRAME {int(batched_prompt_mask_infos['frame_idx'])}")
            print(f"TOTAL: [ {n_tracked + n_filtered + n_not_used} / {n_total} ] | TRACKED: [ {n_tracked} / {n_total} ] | FILTERED: [ {n_filtered} / {n_total} ] | NOT USED: [ {n_not_used} / {n_total} ]")
            print(f"PROCESSING BATCHED PROMPT MASKS -> {[info['prompt_id'] for info in batched_prompt_mask_infos['prompt_infos']]}")
            
            video_predictor.reset_state(inference_state)
            
            # GET PRED MASKLETS AND TOKENS
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
            
            # PROPAGATE IN VIDEO
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                for prompt_idx, prompt_id in enumerate(pred_output_dict['prompt_ids']):
                    pred_output_dict['masklets'][prompt_id][out_frame_idx] = (out_mask_logits[prompt_idx] > 0.0).float()
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=True):
                for prompt_idx, prompt_id in enumerate(pred_output_dict['prompt_ids']):
                    pred_output_dict['masklets'][prompt_id][out_frame_idx] = (out_mask_logits[prompt_idx] > 0.0).float()
            pred_output_dict['masklets'] = {
                prompt_id: torch.cat(pred_output_dict['masklets'][prompt_id], dim=0) for prompt_id in pred_output_dict['prompt_ids']
            }
            
            # GET TOKENS
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

            # SAVE PRED MASKLETS AND TOKENS
            pred_masklet_infos = {}
            for prompt_id in pred_output_dict['prompt_ids']:
                # data format
                pred_masklet_infos[prompt_id] = {
                    "anno_id": prompt_id,
                    "rle": seg_utils.encode_rle_masklet_torch(pred_output_dict['masklets'][prompt_id]),
                    "prompt_type": "SAM2 AMG MASK",
                }
            
            # RESIZE
            pred_output_dict['masklets'] = {
                prompt_id: seg_utils.reshape_masklet(pred_output_dict['masklets'][prompt_id]) for prompt_id in pred_output_dict['prompt_ids']
            }
            
            # COMPUTE PRECISION, RECALL, IOU
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
                # FILTER PROMPT MASKS
                for prompt_mask_info in prompt_mask_infos_expression:
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
                # SAVE PRED MASKLETS AND TOKENS
                with open(os.path.join(sam2_output_dir, "sam2_masklets", video_id, f"{expression_id}", f"{prompt_id:05d}.json"), "w") as f:
                    json.dump(pred_masklet_infos[prompt_id], f, indent=4)
                np.save(os.path.join(sam2_output_dir, "sam2_object_tokens", video_id, f"{expression_id}", f"{prompt_id:05d}.npy"), pred_output_dict['tokens'][prompt_id])
        
        # PRINT
        print("=" * 100)
        print(f"VIDEO {video_id} [ {video_idx} / {len(meta['videos'])} ] | EXP {expression_id} | ELAPSED TIME: {time.time() - start_time:.2f} SEC")
        print(f"TOTAL: [ {n_tracked + n_filtered + n_not_used} / {n_total} ] | TRACKED: [ {n_tracked} / {n_total} ] | FILTERED: [ {n_filtered} / {n_total} ] | NOT USED: [ {n_not_used} / {n_total} ]")
        print("=" * 100)
        not_used_prompt_ids = [info['prompt_id'] for info in prompt_mask_infos_expression if info['status'] == 3]
        filtered_prompt_ids = [info['prompt_id'] for info in prompt_mask_infos_expression if info['status'] == 2]
        tracked_prompt_ids = [info['prompt_id'] for info in prompt_mask_infos_expression if info['status'] == 1]

        # SAVE RUNTIME INFO
        runtime_info[video_id][expression_id] = {
            'time': time.time() - start_time,
            # n tracked, n filtered, n not used, n total
            "n_tracked": n_tracked,
            "n_filtered": n_filtered,
            "n_not_used": n_not_used,
            "n_total": n_total,
            # prompt ids
            "tracked_prompt_ids": tracked_prompt_ids,
            "filtered_prompt_ids": filtered_prompt_ids,
            "not_used_prompt_ids": not_used_prompt_ids,
            "batch_size": args.batch_size,
            "n_frames": n_frames,
            "fps": n_frames / (time.time() - start_time),
        }
        with open(os.path.join(sam2_output_dir, "runtime_info.json"), "w") as f:
            json.dump(runtime_info, f, indent=4)