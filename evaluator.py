import os
import json
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from dataloader import get_loader_dict
from torch.nn.functional import binary_cross_entropy_with_logits
from tools.loss import AlignmentLoss
from tools import metric


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        configs: dict,
        eval_weight_epoch: int,
        device: torch.device,
    ):
        # model
        self.model = model
        self.pred_threshold = configs["eval"]["pred_threshold"]
        self.positive_metric = configs["train"]["positive_metric"]
        self.positive_threshold = configs["train"]["positive_threshold"]
        self.positive_weight = configs["train"]["positive_weight"]
        self.roberta_version = configs["model"]["roberta_version"]
        self.roberta = {
            "tokenizer": AutoTokenizer.from_pretrained(self.roberta_version),
            "model": AutoModel.from_pretrained(self.roberta_version)
        }

        # eval dataset
        self.data_name = configs["dataset"]["valid"]["data_name"]
        self.data_type = configs["dataset"]["valid"]["data_type"]
        self.alignment_weight = configs["train"]["alignment_weight"]
        self.loader_dict = get_loader_dict(configs=configs["dataset"], only_eval=True)
        
        # loss
        self.bce_loss_fn = binary_cross_entropy_with_logits
        self.alignment_loss_fn = AlignmentLoss(
            positive_weight=configs["train"]["positive_weight"],
            temperature=configs["train"]["temperature"],
        )
        
        # etc
        self.device = device
        self.output_dir = configs["results"]["output_dir"]
        self.eval_output_dir = configs["results"]["eval_output_dir"]
        self.eval_weight_epoch = eval_weight_epoch

    @torch.no_grad()
    def evaluate(self):
        # eval
        self.model.eval()
        self.model.to(self.device)
        self.roberta['model'].eval()
        self.roberta['model'].to(self.device)
        
        # metrics
        self.metrics = {
            "total_loss": [],
            "bce_loss": [],
            "alignment_loss": [],
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "tp_score": [], "fp_score": [], "tn_score": [], "fn_score": [],
        }
        
        # eval
        self.pred_dict = {}
        with tqdm(self.loader_dict['valid'], desc=f"EVAL {self.data_type}") as pbar:
            for eval_step, batch in enumerate(pbar):
                # object tokens
                object_tokens = batch['object_tokens'].to(self.device)
                
                # labels
                labels = (
                    batch["labels"][self.positive_metric] > self.positive_threshold
                ).float().to(self.device)
                
                # get lang tokens
                encoded_input = self.roberta['tokenizer'](
                    batch['expression'],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                lang_tokens = self.roberta['model'](**encoded_input)
                def mean_pooling(model_output, attention_mask):
                    token_embeddings = model_output[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                pos_tokens = mean_pooling(lang_tokens, encoded_input['attention_mask']).unsqueeze(1)
                lang_tokens = lang_tokens.last_hidden_state
                batch_size = lang_tokens.shape[0]
                neg_tokens = self.model.negative_token.weight.clone().unsqueeze(0).repeat(batch_size, 1, 1)
                
                # forward
                pred_score, score_tokens = self.model(object_tokens, lang_tokens)
                pred_score = torch.sigmoid(pred_score)
                pred = (pred_score > self.pred_threshold).float()
                
                # loss
                weight = torch.ones_like(labels)
                weight[labels == 1] = self.positive_weight
                bce_loss = self.bce_loss_fn(
                    input=pred_score,
                    target=labels,
                    weight=weight,
                )
                alignment_loss = self.alignment_loss_fn(
                    object_tokens=score_tokens,
                    labels=labels,
                    pos_tokens=pos_tokens,
                    neg_tokens=neg_tokens,
                )
                loss = bce_loss + alignment_loss * self.alignment_weight
                
                # log
                self.metrics["total_loss"].append(loss.item())
                self.metrics["bce_loss"].append(bce_loss.item())
                self.metrics["alignment_loss"].append(alignment_loss.item())
                self.metrics["tp"] += torch.sum((pred == 1) & (labels == 1)).item()
                self.metrics["fp"] += torch.sum((pred == 1) & (labels == 0)).item()
                self.metrics["fn"] += torch.sum((pred == 0) & (labels == 1)).item()
                self.metrics["tn"] += torch.sum((pred == 0) & (labels == 0)).item()
                
                # score
                self.metrics["tp_score"] += pred_score[(pred == 1) & (labels == 1)].cpu().numpy().tolist()
                self.metrics["fp_score"] += pred_score[(pred == 1) & (labels == 0)].cpu().numpy().tolist()
                self.metrics["tn_score"] += pred_score[(pred == 0) & (labels == 0)].cpu().numpy().tolist()
                self.metrics["fn_score"] += pred_score[(pred == 0) & (labels == 1)].cpu().numpy().tolist()

                # pred_dict
                video_id = batch['video_id'][0]
                expression_id = batch['expression_id'][0]
                self.pred_dict[video_id] = self.pred_dict.get(video_id, {})
                self.pred_dict[video_id][expression_id] = {
                    "expression": batch['expression'][0],
                    "anno_ids": batch['anno_ids'][0],
                    "pred": pred.cpu().numpy()[0],
                    "pred_score": pred_score.cpu().numpy()[0],
                    "root_type": batch['root_type'][0],
                    "prompt_type": batch['prompt_type'][0],
                    "sam2_anno_id": batch['sam2_anno_id'][0],
                }
                pbar.set_postfix({"bce loss": f"{bce_loss.item():.4f}"})
        
        self.metrics["total_loss"] = np.mean(self.metrics["total_loss"])
        self.metrics["bce_loss"] = np.mean(self.metrics["bce_loss"])
        self.metrics["alignment_loss"] = np.mean(self.metrics["alignment_loss"])
        self.metrics["accuracy"] = (self.metrics["tp"] + self.metrics["tn"]) / (self.metrics["tp"] + self.metrics["tn"] + self.metrics["fp"] + self.metrics["fn"])
        self.metrics["precision"] = self.metrics["tp"] / (self.metrics["tp"] + self.metrics["fp"] + 1e-6)
        self.metrics["recall"] = self.metrics["tp"] / (self.metrics["tp"] + self.metrics["fn"] + 1e-6)
        self.metrics["f1"] = 2 * self.metrics["precision"] * self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"] + 1e-6)
        self.metrics["tp_score"] = (np.mean(self.metrics["tp_score"]), np.std(self.metrics["tp_score"]))
        self.metrics["fp_score"] = (np.mean(self.metrics["fp_score"]), np.std(self.metrics["fp_score"]))
        self.metrics["tn_score"] = (np.mean(self.metrics["tn_score"]), np.std(self.metrics["tn_score"]))
        self.metrics["fn_score"] = (np.mean(self.metrics["fn_score"]), np.std(self.metrics["fn_score"]))

        print("=" * 50)
        print(f"TOTAL LOSS {self.metrics['total_loss']:.4f} | BCE LOSS {self.metrics['bce_loss']:.4f} | ALIGNMENT LOSS {self.metrics['alignment_loss']:.4f}")
        print(f"ACCURACY: {self.metrics['accuracy']:.4f} PRECISION: {self.metrics['precision']:.4f} RECALL: {self.metrics['recall']:.4f} F1: {self.metrics['f1']:.4f}")
        print(f"TP SCORE: {self.metrics['tp_score']} FP SCORE: {self.metrics['fp_score']} TN SCORE: {self.metrics['tn_score']} FN SCORE: {self.metrics['fn_score']}")

        self.compute_JF_metrics()

        print(f"MEAN J: {self.metrics['mean_J']:.4f} MEAN F: {self.metrics['mean_F']:.4f} MEAN JF: {self.metrics['mean_JF']:.4f}")
        print("=" * 50)
        with open(os.path.join(self.eval_output_dir, f"{self.data_type}_metrics_{self.eval_weight_epoch}epoch.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def compute_JF_metrics(self):
        JF_dict = {}
        Js, Fs, JFs = [], [], []
        for video_idx, video_id in enumerate(self.pred_dict):
            JF_dict[video_id] = {}
            
            # get selected sam2_anno_ids
            self.loader_dict['valid'].dataset.set_video(video_id)
            with tqdm(self.pred_dict[video_id].items(), desc=f"COMPUTE JF METRICS {video_id} [ {video_idx + 1:02d} / {len(self.pred_dict)} ]") as pbar:
                for expression_id, pred_info in pbar:
                    gt_masklet = self.loader_dict['valid'].dataset.get_gt_masklet(video_id, expression_id)
                    pred_masklet = self.loader_dict['valid'].dataset.get_sam2_masklet(
                        video_id=video_id, 
                        expression_id=expression_id, 
                        preds=pred_info["pred"],
                        root_types=pred_info["root_type"],
                        prompt_types=pred_info["prompt_type"],
                        sam2_anno_ids=pred_info["sam2_anno_id"],
                    )
                    
                    if pred_masklet is None:
                        J = 0.0
                        F = 0.0
                        JF = 0.0
                    else:
                        gt_masklet = torch.from_numpy(gt_masklet).float().to(self.device)
                        pred_masklet = torch.from_numpy(pred_masklet).float().to(self.device)
                        J = float(self.compute_J(pred_masklet, gt_masklet))
                        F = float(self.compute_F(pred_masklet, gt_masklet))
                        JF = (J + F) / 2
                    
                    JF_dict[video_id][expression_id] = {
                        "expression": pred_info["expression"],
                        "J": J,
                        "F": F,
                        "JF": JF,
                    }
                    Js.append(J)
                    Fs.append(F)
                    JFs.append(JF)
                    pbar.set_postfix({
                        "J": f"{np.mean(Js):.4f}",
                        "F": f"{np.mean(Fs):.4f}",
                        "JF": f"{np.mean(JFs):.4f}",
                    })
        
        self.metrics["mean_J"] = np.mean(Js)
        self.metrics["mean_F"] = np.mean(Fs)
        self.metrics["mean_JF"] = np.mean(JFs)
        
        with open(os.path.join(self.eval_output_dir, f"{self.data_type}_JF_metrics_{self.eval_weight_epoch}epoch.json"), "w") as f:
            json.dump(JF_dict, f, indent=4)

    def compute_J(self, pred_masklet, gt_masklet):
        Js = []
        n_frames = pred_masklet.shape[0]
        for i in range(n_frames):
            intersection = (pred_masklet[i] * gt_masklet[i]).sum().item()
            union = (pred_masklet[i] + gt_masklet[i]).sum().item() - intersection
            if union == 0:
                Js.append(1.0)
            else:
                Js.append(intersection / union)
        return np.mean(Js)

    def compute_F(self, pred_masklet, gt_masklet):
        tp = (pred_masklet * gt_masklet).sum().item()
        fp = ((1 - gt_masklet) * pred_masklet).sum().item()
        fn = (gt_masklet * (1 - pred_masklet)).sum().item()
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)
