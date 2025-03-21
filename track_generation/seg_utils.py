import os
import numpy as np
import torch
from pycocotools import mask as mask_utils
from PIL import Image

# get masklet from mask_dict
def get_masklet(anno_id, mask_dict):
    rle_masklet = mask_dict[str(anno_id)]
    masklet = []
    h, w = 0, 0
    for rle_mask in rle_masklet:
        if isinstance(rle_mask, dict):
            if isinstance(rle_mask['counts'], bytes):
                rle_mask['counts'] = rle_mask['counts'].encode('utf-8')
            mask = mask_utils.decode(rle_mask)
            h, w = mask.shape
            masklet.append(mask)
        else:
            masklet.append(None)
    for mask_idx in range(len(masklet)):
        if masklet[mask_idx] is None:
            masklet[mask_idx] = np.zeros((h, w), dtype=np.uint8)
    masklet = np.stack(masklet, axis=0)
    masklet = torch.from_numpy(masklet).float()
    return masklet

# get masklets from ytbvos format
def get_masklets_ytbvos(masklet_dir, reshape=False):
    mask_paths = os.listdir(masklet_dir)
    mask_paths = sorted(mask_paths)
    masklets = {}
    for mask_path in mask_paths:
        masks = Image.open(os.path.join(masklet_dir, mask_path))
        masks = masks.convert("P")
        masks = np.array(masks)
        for obj_id in range(1, 256):
            mask = (masks == obj_id).astype(np.uint8)
            masklets[str(obj_id)] = masklets.get(str(obj_id), []) + [mask]
    del_obj_ids = []
    for obj_id in masklets:
        masklets[obj_id] = torch.from_numpy(np.stack(masklets[obj_id], axis=0)).float()
        if masklets[obj_id].sum() == 0:
            del_obj_ids.append(obj_id)
        if reshape:
            masklets[obj_id] = reshape_masklet(masklets[obj_id])
    for del_obj_id in del_obj_ids:
        del masklets[del_obj_id]
    return masklets

# get masklets from meta data
def get_masklets(video_id, meta, mask_dict):
    video_meta = meta['videos'][video_id]
    masklets = {}
    for _, expression_meta in video_meta["expressions"].items():
        for anno_id in expression_meta["anno_id"]:
            if anno_id in masklets:
                continue
            else:
                masklets[anno_id] = get_masklet(anno_id, mask_dict)
    return masklets

# decode rle mask
def decode_rle_mask(rle_mask):
    if isinstance(rle_mask['counts'], bytes):
        rle_mask['counts'] = rle_mask['counts'].encode('utf-8')
    return mask_utils.decode(rle_mask)

# decode rle masklet
def decode_rle_masklet(rle_masklet):
    masklet = []
    for rle_mask in rle_masklet:
        masklet.append(decode_rle_mask(rle_mask))
    masklet = np.stack(masklet, axis=0)
    return masklet

# encode rle masklet
def encode_rle_masklet(masks):
    '''
    args:
        mask: np.ndarray, (N, H, W), {0, 1}
    return:
        encoded_rle: dict, with keys: ['size', 'counts']
    '''
    encoded_rle = []
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        encoded_rle.append(rle)
    return encoded_rle

# encode rle masklet torch
def encode_rle_masklet_torch(masks):
    '''
    args:
        mask: torch.Tensor, (N, H, W), {0, 1}
    return:
        encoded_rle: dict, with keys: ['size', 'counts']
    '''
    masks = masks.cpu().numpy().astype(np.uint8)
    encoded_rle = []
    for mask in masks:
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        encoded_rle.append(rle)
    return encoded_rle

# compute masklet iou
@torch.no_grad()
def compute_masklet_iou(maskletA, maskletB, device):
    '''
    args:
        maskletA: torch.Tensor, (N, H, W), {0, 1}
        maskletB: torch.Tensor, (N, H, W), {0, 1}
    return:
        iou: float
    '''
    maskletA = maskletA.to(device)
    maskletB = maskletB.to(device)
    intersection = torch.sum(maskletA * maskletB).item()
    union = torch.sum(maskletA + maskletB).item() - intersection
    if union == 0:
        return 1.0
    iou = intersection / union
    return iou

# compute mask iou
@torch.no_grad()
def compute_mask_iou(maskA, maskB):
    '''
    args:
        maskA: torch.Tensor, (H, W), {0, 1}
        maskB: torch.Tensor, (H, W), {0, 1}
    return:
        iou: float
    '''
    intersection = torch.sum(maskA * maskB).item()
    union = torch.sum(maskA + maskB).item() - intersection
    if union == 0.0:
        return 1.0
    iou = intersection / union
    return iou

# reshape masklet torch
def reshape_masklet(masklet, target_shape=None):
    '''
    args:
        masklet: torch.Tensor, (N, H, W), {0, 1}
        target_shape: tuple or None, (H', W') or None
    return:
        reshaped_masklet: torch.Tensor, (N, H', W'), {0, 1}
    '''
    if target_shape is None:
        ori_h, ori_w = masklet.shape[1:]
        new_h, new_w = (540, 960) if ori_h < ori_w else (960, 540)
    else:
        new_h, new_w = target_shape
    masklet = torch.nn.functional.interpolate(masklet.unsqueeze(0), size=(new_h, new_w), mode='bilinear') > 0.5
    masklet = masklet.squeeze(0).float()
    return masklet

# get area thresholds
def get_area_threshs_from_sample(prompt_mask_infos, n_area_bins, n_prompts):
    area_threshs = []
    step = n_prompts // n_area_bins
    step = max(step, 1)
    for prompt_frame_idx in prompt_mask_infos:
        for prompt_mask_info in prompt_mask_infos[prompt_frame_idx]:
            area_threshs.append(prompt_mask_info['area_ratio'])
    area_threshs = sorted(area_threshs, reverse=True)
    area_threshs = area_threshs[step - 1::step]
    area_threshs.append(0.0)
    return area_threshs

