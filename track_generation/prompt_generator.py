import torch
import numpy as np
from PIL import Image
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class PromptGenerator:
    def __init__(
        self,
        # grounding dino
        grounding_dino_cfg: str,
        grounding_dino_ckpt: str,
        # sam2
        sam2: torch.nn.Module = None,
        sam2_cfg: str = None,
        sam2_ckpt: str = None,
        # etc
        device: str = "cpu",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        assert sam2 is not None or (sam2_cfg is not None and sam2_ckpt is not None), "sam2 or sam2_cfg and sam2_ckpt should be provided"
        # configs
        self.grounding_dino_cfg = grounding_dino_cfg
        self.grounding_dino_ckpt = grounding_dino_ckpt
        self.sam2_cfg = sam2_cfg
        self.sam2_ckpt = sam2_ckpt
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        # load grounding dino model
        self.load_grounding_dino()
        # load sam2 model
        self.load_sam2(sam2)

    def load_grounding_dino(self):
        print("LOAD GROUNDING DINO MODEL")
        args = SLConfig.fromfile(self.grounding_dino_cfg)
        args.device = self.device
        self.grounding_dino = build_model(args)
        checkpoint = torch.load(self.grounding_dino_ckpt, map_location="cpu")
        load_res = self.grounding_dino.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = self.grounding_dino.eval()
        self.grounding_dino.to(self.device)
        self.transform_gd = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def load_sam2(self, sam2=None):
        print("LOAD SAM2 MODEL")
        if sam2 is None:
            sam2 = build_sam2(self.sam2_cfg, self.sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2)

    def generate_prompts(
        self,
        raw_image: Image.Image,
        raw_texts: list[str],
    ):
        '''
        args:
            raw_image: PIL.Image.Image, shape=(H, W, 3)
            raw_texts: list[str]
        return:
            prompt_masks: dict
                "expression": str
                "tokenized": list[str]
                "preds": list[dict]
                    "phrase": str
                    "bbox": np.ndarray, shape=(4,)
                    "token_score": list[float]
                    "sam2_mask": np.ndarray, shape=(H, W)
                    "mask_score": float
                    "stability_score": float
        '''
        self.raw_image = raw_image
        self.raw_texts = raw_texts
        # get bbox prompts
        outputs = self.get_bbox_prompts()
        # set image
        image_np = np.array(self.raw_image)
        self.sam2_predictor.set_image(image_np)
        for text_id in outputs:
            bboxes = []
            if len(outputs[text_id]['preds']) == 0:
                continue
            for bbox_info in outputs[text_id]['preds']:
                bbox = bbox_info['bbox']
                bboxes.append(bbox)
            bboxes = np.stack(bboxes, axis=0)
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bboxes,
                multimask_output=False,
            )
            if masks.ndim >= 4:
                masks = masks[:, 0]
                scores = scores[:, 0]
                logits = logits[:, 0]
            for bbox_idx, (mask, score, logit) in enumerate(zip(masks, scores, logits)):
                stability_score = self.get_stability_score(logit)
                outputs[text_id]['preds'][bbox_idx].update({
                    'sam2_mask': mask,
                    'mask_score': float(score),
                    'stability_score': float(stability_score),
                })
        return outputs

    @torch.no_grad()
    def get_bbox_prompts(self):
        # load image
        image = self.load_image_gd()
        outputs = {}
        for text_idx, raw_text in enumerate(self.raw_texts):
            # text preprocess
            raw_text = raw_text.lower()
            raw_text = raw_text.strip()
            if not raw_text.endswith("."):
                raw_text += "."
            # inference
            outputs_per_text = self.grounding_dino(image[None], captions=[raw_text])
            logits = outputs_per_text["pred_logits"].sigmoid()[0]
            boxes = outputs_per_text["pred_boxes"][0]
            # filtering
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
            logits_filt = logits_filt[filt_mask] # [num_filt, 256]
            boxes_filt = boxes_filt[filt_mask] # [num_filt, 4]
            # get phrase
            tokenized = self.grounding_dino.tokenizer(raw_text)
            outputs[str(text_idx)] = {
                "expression": raw_text,
                "tokenized": [self.grounding_dino.tokenizer.decode([input_id]) for input_id in tokenized['input_ids']],
                "preds": []
            }
            n_tokens = len(outputs[str(text_idx)]['tokenized'])
            for logit, bbox in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, self.grounding_dino.tokenizer)
                # from 0..1 to 0..W, 0..H
                bbox = bbox * self.box_scale
                # from xywh to xyxy
                bbox[:2] -= bbox[2:] / 2
                bbox[2:] += bbox[:2]
                outputs[str(text_idx)]['preds'].append({
                    "phrase": pred_phrase,
                    "bbox": bbox.cpu().numpy(),
                    "token_score": logit[:n_tokens].cpu().numpy().tolist(),
                })
        return outputs

    def load_image_gd(self):
        W, H = self.raw_image.size
        self.box_scale = torch.Tensor([W, H, W, H])
        image, _ = self.transform_gd(self.raw_image, None)
        return image.to(self.device)

    def get_stability_score(self, logit, mask_threshold=0.0, threshold_offset=1.0):
        '''
        args:
            logit: np.ndarray, shape=(H, W)
        return:
            stability_score: float
        '''
        intersection = (
            (logit > (mask_threshold + threshold_offset))
            .sum(-1, dtype=np.int16)
            .sum(-1, dtype=np.int32)
        )
        union = (
            (logit > (mask_threshold - threshold_offset))
            .sum(-1, dtype=np.int16)
            .sum(-1, dtype=np.int32)
        )
        return intersection / union