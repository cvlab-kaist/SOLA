import os
import json
import numpy as np
import random
import torch
import pycocotools.mask as mask_utils
from PIL import Image

from torch.utils.data import Dataset, DataLoader


def get_loader_dict(
    configs: dict, 
    only_eval: bool=False,
    only_inference: bool=False,
) -> dict:
    assert not (only_eval and only_inference), "only_eval and only_inference cannot be True at the same time"
    
    for split in ["train", "valid", "test"]:
        configs[split]["data_root"] = configs["data_root"]
        configs[split]["track_root"] = configs["track_root"]
        configs[split]["num_workers"] = configs["num_workers"]

    if only_eval:
        return {
            "valid": get_loader(configs["valid"]),
        }
    elif only_inference:
        return {
            "test": get_loader(configs["test"]),
        }
    else:
        return {
            "train": get_loader(configs["train"]),
            "valid": get_loader(configs["valid"]),
        }

def get_loader(configs: dict) -> DataLoader:
    return DataLoader(
        dataset=AlignDataset(configs=configs),
        batch_size=configs["batch_size"],
        shuffle=(configs["data_type"]=="train"),
        num_workers=configs["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

def collate_fn(batch):
    if batch == [None]:
        raise ValueError("batch is None")

    no_gt = batch[0]['labels'] is None
    if no_gt:
        batch =  {
            "video_id": [sample['video_id'] for sample in batch],
            "frames": [sample['frames'] for sample in batch],
            "expression_id": [sample['expression_id'] for sample in batch],
            "expression": [sample['expression'] for sample in batch],
            "anno_ids": [sample['anno_ids'] for sample in batch],
            "object_tokens": torch.stack([sample['object_tokens'] for sample in batch], axis=0),
            "root_type": [sample['root_type'] for sample in batch],
            "prompt_type": [sample['prompt_type'] for sample in batch],
            "sam2_anno_id": [sample['sam2_anno_id'] for sample in batch],
            "gt_anno_id": [sample['gt_anno_id'] for sample in batch],
        }
    else:
        batch = {
            "video_id": [sample['video_id'] for sample in batch],
            "frames": [sample['frames'] for sample in batch],
            "expression_id": [sample['expression_id'] for sample in batch],
            "expression": [sample['expression'] for sample in batch],
            "anno_ids": [sample['anno_ids'] for sample in batch],
            "object_tokens": torch.stack([sample['object_tokens'] for sample in batch], axis=0),
            "labels": {
                "iou": torch.stack([sample['labels']['iou'] for sample in batch], axis=0),
                "recall": torch.stack([sample['labels']['recall'] for sample in batch], axis=0),
                "precision": torch.stack([sample['labels']['precision'] for sample in batch], axis=0),
            },
            "root_type": [sample['root_type'] for sample in batch],
            "prompt_type": [sample['prompt_type'] for sample in batch],
            "sam2_anno_id": [sample['sam2_anno_id'] for sample in batch],
            "gt_anno_id": [sample['gt_anno_id'] for sample in batch],
        }
    
    return batch

class AlignDataset(Dataset):
    def __init__(self, configs: dict) -> None:
        super().__init__()
        self.data_name = configs["data_name"]
        self.data_type = configs["data_type"]
        self.data_root = configs["data_root"]
        self.track_root = configs["track_root"]
        self.sam2_output_dirs = configs["sam2_output_dirs"].split(",")
        self.video_id = None
        self.NO_OBJECT_ID = -1
        # load data
        self.load_data()
    
    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        # get meta data
        meta = self.meta_list[idx]
        video_id = meta['video_id']
        expression_id = meta['expression_id']
        expression = meta['expression']
        anno_ids = meta['anno_ids']
        frames = meta['frames']
        # load sam2 outputs
        object_tokens = []
        labels = {
            "iou": [],
            "recall": [],
            "precision": [],
        }
        root_type, prompt_type, sam2_anno_ids, gt_anno_ids = [], [], [], []
        for sam2_output_dir in self.sam2_output_dirs:
            sam2_output_dir = os.path.join(self.track_root, sam2_output_dir)
            # get masklet and token directories
            if "gdino" in sam2_output_dir:
                sam2_masklet_dir = os.path.join(sam2_output_dir, self.data_name, self.data_type, "sam2_masklets", video_id, expression_id)
                sam2_token_dir = os.path.join(sam2_output_dir, self.data_name, self.data_type, "sam2_object_tokens", video_id, expression_id)
            else:
                sam2_masklet_dir = os.path.join(sam2_output_dir, self.data_name, self.data_type, "sam2_masklets", video_id)
                sam2_token_dir = os.path.join(sam2_output_dir, self.data_name, self.data_type, "sam2_object_tokens", video_id)
            # get masklet paths
            sam2_masklet_paths = os.listdir(sam2_masklet_dir)
            sam2_masklet_paths.sort()
            # get masklets
            gt_anno_id_cnt = {}
            gt_anno_id_max_cnt = {}
            for sam2_masklet_path in sam2_masklet_paths:
                # get masklet
                with open(os.path.join(sam2_masklet_dir, sam2_masklet_path), "r") as f:
                    sam2_masklet_info = json.load(f)
                sam2_anno_id = sam2_masklet_info['anno_id']
                # get labels if gt_anno_id exists
                iou, recall, precision, gt_anno_id = 0.0, 0.0, 0.0, self.NO_OBJECT_ID
                if anno_ids[0] >= 0: # gt anno_ids
                    for anno_id in anno_ids:
                        miou = sam2_masklet_info.get('iou', {}).get(str(anno_id), 0.0)
                        mprecision = sam2_masklet_info.get('precision', {}).get(str(anno_id), 0.0)
                        mrecall = sam2_masklet_info.get('recall', {}).get(str(anno_id), 0.0)
                        if miou > iou:
                            iou = miou
                            recall = mrecall
                            precision = mprecision
                            gt_anno_id = anno_id
                # metrics
                labels['iou'].append(iou)
                labels['recall'].append(recall)
                labels['precision'].append(precision)
                # anno_ids
                gt_anno_ids.append(gt_anno_id)
                sam2_anno_ids.append(sam2_anno_id)
                # get root_type and prompt_type
                root_type.append(os.path.basename(sam2_output_dir))
                prompt_type.append(sam2_masklet_info['prompt_type'])
                # get object token
                object_token = np.load(os.path.join(sam2_token_dir, f"{sam2_anno_id:05d}.npy"))
                object_tokens.append(torch.from_numpy(object_token))
        # check object tokens
        assert len(object_tokens) > 0, f"object_tokens is empty"
        # object tokens
        object_tokens = torch.stack(object_tokens, axis=0)
        # labels
        if anno_ids[0] < 0: # gt anno id does not exist
            assert (
                (self.data_name == "mevis" and self.data_type == "valid") or
                (self.data_name == "ref-ytbvos" and self.data_type == "valid")
            ), f"Invalid data_name: {self.data_name}, data_type: {self.data_type}"
            labels = None
        else: # gt anno id exists
            assert (
                (self.data_name == "mevis" and self.data_type in ["train", "valid_u"]) or
                (self.data_name == "ref-ytbvos" and self.data_type == "train") or
                (self.data_name == "ref-davis" and self.data_type in ["train", "valid"])
            ), f"Invalid data_name: {self.data_name}, data_type: {self.data_type}"
            labels['iou'] = torch.tensor(labels['iou'])
            labels['recall'] = torch.tensor(labels['recall'])
            labels['precision'] = torch.tensor(labels['precision'])
        return {
            # video info
            "video_id": video_id,
            "frames": frames,
            # expression info
            "expression_id": expression_id,
            "expression": expression,
            "anno_ids": anno_ids,
            # sam2 outputs
            "object_tokens": object_tokens,
            "labels": labels,
            # root_type and prompt_type
            "root_type": root_type,
            "prompt_type": prompt_type,
            "sam2_anno_id": sam2_anno_ids,
            "gt_anno_id": gt_anno_ids,
        }

    def load_data(self):
        if self.data_name == "mevis":
            # load meta data
            with open(os.path.join(self.data_root, self.data_name, self.data_type, "meta_expressions.json"), "r") as f:
                self.meta = json.load(f)
            # load mask data
            if self.data_type in ["train", "valid_u"]:
                with open(os.path.join(self.data_root, self.data_name, self.data_type, "mask_dict.json"), "r") as f:
                    self.mask_dict = json.load(f)
            # load samples
            self.meta_list = []
            for video_id, video_meta in self.meta["videos"].items():
                for expression_id, expression_meta in video_meta["expressions"].items():
                    self.meta_list.append({
                        "video_id": video_id,
                        "expression_id": expression_id,
                        "expression": expression_meta["exp"],
                        "anno_ids": expression_meta.get("anno_id", [self.NO_OBJECT_ID]),
                        "frames": video_meta["frames"],
                    })
        elif self.data_name in ["ref-ytbvos", "ref-davis"]:
            # load meta data
            with open(os.path.join(self.data_root, self.data_name, "meta_expressions", self.data_type, "meta_expressions.json"), "r") as f:
                self.meta = json.load(f)
            # load samples
            self.meta_list = []
            for video_id, video_meta in self.meta["videos"].items():
                for expression_id, expression_meta in video_meta["expressions"].items():
                    self.meta_list.append({
                        "video_id": video_id,
                        "expression_id": expression_id,
                        "expression": expression_meta["exp"],
                        "anno_ids": [int(expression_meta.get("obj_id", self.NO_OBJECT_ID))],
                        "frames": video_meta["frames"],
                    })
        else:
            raise ValueError(f"Invalid data_name: {self.data_name}")

    # evaluate
    def set_video(self, video_id):
        if self.video_id is None:
            self.video_id = video_id
            self.load_gt_masklet(video_id)
        elif self.video_id != video_id:
            self.video_id = video_id
            self.load_gt_masklet(video_id)
        else:
            raise NotImplementedError
    
    def load_gt_masklet(self, video_id):
        self.cached_gt_masklet = {}
        if self.data_name == "mevis":
            for _, expression_meta in self.meta['videos'][video_id]['expressions'].items():
                gt_anno_ids = expression_meta['anno_id']
                for gt_anno_id in gt_anno_ids:
                    gt_anno_id = str(gt_anno_id)
                    if gt_anno_id not in self.cached_gt_masklet:
                        self.cached_gt_masklet[gt_anno_id] = self.rle_masklet_decode(self.mask_dict[gt_anno_id])
        elif self.data_name == "ref-davis":
            anno_dir = os.path.join(self.data_root, "ref-davis", self.data_type, "Annotations", video_id)
            frames = sorted(os.listdir(anno_dir))
            anno_mask_image = Image.open(os.path.join(anno_dir, "00000.png")).convert("P")
            W, H = anno_mask_image.size[0], anno_mask_image.size[1]
            masklet = np.zeros((len(frames), H, W), dtype=np.uint8)
            object_ids = np.unique(np.array(anno_mask_image))
            object_ids = object_ids[(object_ids != 0) & (object_ids != 255)]
            for object_id in object_ids:
                for frame_idx, frame in enumerate(frames):
                    obj_mask_image = Image.open(os.path.join(anno_dir, frame)).convert("P")
                    mask = np.array(obj_mask_image) == object_id
                    masklet[frame_idx] = mask
                if object_id not in self.cached_gt_masklet:
                    self.cached_gt_masklet[object_id] = masklet
        else:
            raise ValueError(f"Invalid data_name: {self.data_name}")

    def get_gt_masklet(self, video_id, expression_id):
        assert self.video_id == video_id, f"video_id is not set: {self.video_id} != {video_id}"
        if self.data_name == "mevis" or self.data_name == "ref-davis":
            expression_meta = self.meta['videos'][video_id]['expressions'][expression_id]
            if self.data_name == "ref-davis":
                gt_anno_ids = expression_meta['obj_id']
            else:
                gt_anno_ids = expression_meta['anno_id']
            merged_masklet = None
            for gt_anno_id in gt_anno_ids:
                gt_anno_id = str(gt_anno_id)
                if gt_anno_id in self.cached_gt_masklet:
                    masklet = self.cached_gt_masklet[gt_anno_id]
                elif int(gt_anno_id) in self.cached_gt_masklet:
                    masklet = self.cached_gt_masklet[int(gt_anno_id)]
                else:
                    masklet = self.rle_masklet_decode(self.mask_dict[str(gt_anno_id)])
                if merged_masklet is None:
                    merged_masklet = masklet
                else:
                    merged_masklet = np.logical_or(merged_masklet, masklet)
            return merged_masklet
        elif self.data_name == "ref-ytbvos":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid data_name: {self.data_name}")

    def get_sam2_masklet(
        self, 
        video_id: str, 
        expression_id: str, 
        preds: np.ndarray,
        root_types: list,
        prompt_types: list,
        sam2_anno_ids: list,
    ) -> np.ndarray:
        merged_masklet = None
        sam2_anno_idx = 0
        for sam2_output_dir in self.sam2_output_dirs:
            sam2_output_dir = os.path.join(self.track_root, sam2_output_dir)
            # get masklet and token directories
            if "gdino" in sam2_output_dir:
                sam2_masklet_dir = os.path.join(sam2_output_dir, self.data_name, self.data_type, "sam2_masklets", video_id, expression_id)
            else:
                sam2_masklet_dir = os.path.join(sam2_output_dir, self.data_name, self.data_type, "sam2_masklets", video_id)
            # get masklet paths
            sam2_masklet_paths = os.listdir(sam2_masklet_dir)
            sam2_masklet_paths.sort()
            for sam2_masklet_path in sam2_masklet_paths:
                if preds[sam2_anno_idx] < 1 and merged_masklet is not None:
                    sam2_anno_idx += 1
                    continue
                # get masklet
                with open(os.path.join(sam2_masklet_dir, sam2_masklet_path), "r") as f:
                    sam2_masklet_info = json.load(f)
                root_type = root_types[sam2_anno_idx]
                prompt_type = prompt_types[sam2_anno_idx]
                sam2_anno_id = sam2_anno_ids[sam2_anno_idx]
                assert root_type == os.path.basename(sam2_output_dir), f"Invalid root_type: {root_type} != {os.path.basename(sam2_output_dir)}"
                assert prompt_type == sam2_masklet_info['prompt_type'], f"Invalid prompt_type: {prompt_type} != {sam2_masklet_path['prompt_type']}"
                assert sam2_anno_id == sam2_masklet_info['anno_id'], f"Invalid sam2_anno_id: {sam2_anno_id} != {sam2_masklet_path['anno_id']}"
                if preds[sam2_anno_idx] > 0:
                    sam2_masklet = self.rle_masklet_decode(sam2_masklet_info['rle'])
                    if merged_masklet is None:
                        merged_masklet = sam2_masklet
                    else:
                        merged_masklet = np.logical_or(merged_masklet, sam2_masklet)
                else:
                    if merged_masklet is None:
                        h, w = sam2_masklet_info["rle"][0]["size"]
                        t = len(sam2_masklet_info["rle"])
                        merged_masklet = np.zeros((t, h, w), dtype=np.uint8)
                sam2_anno_idx += 1
        return merged_masklet

    def rle_masklet_decode(self, rle_masklet):
        masklet = []
        h, w = 0,0
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
        return masklet

    def get_frames(self, video_id):
        return self.meta['videos'][video_id]['frames']
