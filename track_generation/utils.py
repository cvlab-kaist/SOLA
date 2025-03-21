import pycocotools.mask as mask_utils
import numpy as np
import torch
import cv2

# fn : encode rle mask fn
def encode_rle_mask(mask):
    '''
    args:
        mask: np.array or torch.Tensor, shape=(H, W)
    return:
        rle_mask: dict
        - "size": list
        - "counts": str
    '''
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
    elif isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().astype(np.uint8)
    mask = np.asfortranarray(mask)
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# fn : encode rle masklet torch fn
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

# fn : decode rle mask fn
def decode_rle_mask(rle_mask, reshape_size=False):
    '''
    args:
        rle_mask: dict
        - "size": list
        - "counts": str or bytes
    return:
        mask: np.array, shape=(H, W)
    '''
    if not isinstance(rle_mask, dict):
        return None
    if isinstance(rle_mask["counts"], bytes):
        rle_mask["counts"] = rle_mask["counts"].encode("utf-8")
    mask = mask_utils.decode(rle_mask)
    if reshape_size:
        h, w = rle_mask["size"]
        h, w = (960, 540) if h > w else (540, 960)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8)
    return mask

# fn : compute mask iou fn
@torch.no_grad()
def compute_mask_iou_torch(maskA, maskB):
    '''
    args:
        maskA: torch.Tensor, shape=(H, W)
        maskB: torch.Tensor, shape=(H, W)
    return:
        iou: float
    '''
    intersection = (maskA * maskB).sum().item()
    union = maskA.sum().item() + maskB.sum().item() - intersection
    return intersection / union

# fn : reshape masklet torch fn
@torch.no_grad()
def reshape_masklet_torch(masklet, target_shape=None):
    '''
    args:
        masklet: torch.Tensor, shape=(N, H, W)
        target_shape: tuple, (H, W)
    '''
    if target_shape is None:
        ori_h, ori_w = masklet.shape[1:]
        new_h, new_w = (960, 540) if ori_h > ori_w else (540, 960)
    else:
        new_h, new_w = target_shape
    masklet = torch.nn.functional.interpolate(masklet.unsqueeze(0), size=(new_h, new_w), mode='bilinear') > 0.5
    masklet = masklet.squeeze(0).float()
    return masklet

# fn : put text box fn
def put_text_box(img, text, top_left, bottom_right, color, font_scale=1, thickness=1):
    '''
    args:
        img: np.array, shape=(H, W, 3), dtype=np.uint8
        text: str
        top_left: tuple, (x, y)
        bottom_right: tuple, (x, y)
        color: tuple, (r, g, b)
    return:
        img: np.array, shape=(H, W, 3), dtype=np.uint8
    '''
    if text.strip() == "":
        text = "[NONE]"
    text_size, _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_ITALIC, fontScale=font_scale, thickness=thickness)
    text_w, text_h = text_size
    x0, y0 = top_left
    x1, y1 = bottom_right
    offset_h = 10
    if y0 - text_h < 0:
        if y1 + text_h > img.shape[0]:
            pt1 = (x0, y0)
            pt2 = (x0 + text_w, y0 + text_h)
            org = (x0, y0 + text_h)
        else:
            pt1 = (x0, y1)
            pt2 = (x0 + text_w, y1 + text_h)
            org = (x0, y1 + text_h)
    else:
        pt1 = (x0, y0 - text_h)
        pt2 = (x0 + text_w, y0)
        org = (x0, y0)
    cv2.rectangle(img, pt1, pt2, color, -1)
    cv2.putText(img, text, org, cv2.FONT_ITALIC, font_scale, (0, 0, 0), thickness)
    return img


@torch.no_grad()
def compute_mask_metrics(pred_masks, gt_masks, reduction="mean"):
    '''
    args:
        pred_masks: torch.Tensor, shape=(T, H, W)
        gt_masks  : torch.Tensor, shape=(T, H, W)
    return:
        precision: float
        recall   : float
        iou      : float
    '''
    T = pred_masks.shape[0]
    precision = torch.zeros(T).float()
    recall = torch.zeros(T).float()
    iou = torch.zeros(T).float()
    for t in range(T):
        intersection = (pred_masks[t] * gt_masks[t]).sum().item()
        union = (pred_masks[t] + gt_masks[t]).sum().item() - intersection
        n_pred = pred_masks[t].sum().item()
        n_gt = gt_masks[t].sum().item()
        # compute iou
        if union == 0:
            iou[t] = 1.0
        else:
            iou[t] = intersection / union
        # compute precision and recall
        if n_pred == 0 and n_gt == 0:
            precision[t] = 1.0
            recall[t] = 1.0
        elif n_pred == 0 and n_gt > 0:
            precision[t] = 1.0
            recall[t] = 0.0
        elif n_pred > 0 and n_gt == 0:
            precision[t] = 0.0
            recall[t] = 1.0
        else:
            precision[t] = intersection / n_pred
            recall[t] = intersection / n_gt
    if reduction == "mean":
        return precision.mean(), recall.mean(), iou.mean()
    elif reduction == "none":
        return precision, recall, iou
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

# FN: COMPUTE PARTNESS
@torch.no_grad()
def compute_P(part_masks, full_mask):
    '''
    args:
        part_masks: torch.Tensor, shape=(N, H, W)
        full_mask: torch.Tensor, shape=(H, W)
    return:
        P: torch.Tensor, shape=(N)
    '''
    N = part_masks.shape[0]
    part_masks = part_masks.reshape(N, -1)
    full_mask = full_mask.reshape(-1, 1)
    intersection = part_masks @ full_mask # [N, 1]
    n_part = part_masks.sum(dim=1, keepdim=True) # [N, 1]
    partness = intersection / n_part
    return partness.squeeze(1)