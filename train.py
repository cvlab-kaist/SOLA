import os
import yaml
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import warnings

import torch
from transformers import AutoTokenizer, AutoModel

from module.module import LanguageAlignedTrackSelectionModule
from dataloader import get_loader_dict
from torch.nn.functional import binary_cross_entropy_with_logits
from tools.loss import AlignmentLoss
from tools import metric


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(configs):
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MODEL
    track_selection_module = LanguageAlignedTrackSelectionModule(configs['model']).to(device)
    
    # TEXT ENCODER
    tokenizer = AutoTokenizer.from_pretrained(configs['model']['roberta_version'])
    roberta = AutoModel.from_pretrained(configs['model']['roberta_version']).to(device)
    
    # DATALOADER
    loader_dict = get_loader_dict(configs["dataset"])
    
    # LOSS FUNCTION
    alignment_loss_fn = AlignmentLoss(
        positive_weight=configs["train"]["positive_weight"],
        temperature=configs["train"]["temperature"],
    )
    
    # OPTIMIZER
    optimizer = torch.optim.AdamW(
        params=[
            {"params": track_selection_module.parameters()},
        ],
        lr=configs["train"]["lr"],
    )

    # LR SCHEDULER
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=configs["train"]["lr_factor"],
        patience=configs["train"]["lr_patience"],
    )
    
    # TRAIN LOOP
    train_step = 0
    for epoch in range(configs["train"]["n_epochs"]):
        track_selection_module.train()
        roberta.eval()
        train_loss = {
            "total": [],
            "bce": [],
            "alignment": [],
        }
        with tqdm(loader_dict["train"], desc=f"EPOCH [{epoch + 1} / {configs['train']['n_epochs']}]") as pbar:
            for batch in pbar:
                # object tokens
                object_tokens = batch["object_tokens"].to(device)
                
                # labels
                labels = (
                    batch["labels"][configs["train"]["positive_metric"]] > configs["train"]["positive_threshold"]
                ).float().to(device)
                
                # get lang tokens
                encoded_input = tokenizer.batch_encode_plus(
                    batch['expression'], 
                    padding="longest", 
                    return_tensors="pt"
                ).to(device)
                lang_tokens = roberta(**encoded_input)
                def mean_pooling(model_output, attention_mask):
                    token_embeddings = model_output[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                pos_tokens = mean_pooling(lang_tokens, encoded_input['attention_mask']).unsqueeze(1)
                lang_tokens = lang_tokens.last_hidden_state
                neg_tokens = track_selection_module.negative_token.weight.clone().unsqueeze(0).repeat(batch_size, 1, 1)
                
                # forward
                score_logits, score_tokens = track_selection_module(object_tokens, lang_tokens)
                
                # binary cross entropy loss
                weight = torch.ones_like(labels)
                weight[labels > 0] = configs["train"]["positive_weight"]
                bce_loss = binary_cross_entropy_with_logits(
                    input=score_logits,
                    target=labels,
                    weight=weight,
                )
                # alignment loss
                alignment_loss = alignment_loss_fn(
                    object_tokens=score_tokens,
                    labels=labels,
                    pos_tokens=pos_tokens,
                    neg_tokens=neg_tokens,
                )
                # total loss
                loss = bce_loss + alignment_loss * configs["train"]["alignment_weight"]
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                
                # gradient clipping
                grad_norm_dict = track_selection_module.get_grad_norm_dict()
                if configs["train"]["grad_clip_norm"] > 0 and grad_norm_dict["total_grad_norm"] > configs["train"]["grad_clip_norm"]:
                    torch.nn.utils.clip_grad_norm_(track_selection_module.parameters(), configs["train"]["grad_clip_norm"])
                
                # optimizer step
                optimizer.step()
                
                # log
                train_loss["total"].append(loss.item())
                train_loss["bce"].append(bce_loss.item())
                train_loss["alignment"].append(alignment_loss.item())
                train_log = {
                    "total": loss.item(),
                    "bce": bce_loss.item(),
                    "alignment": alignment_loss.item(),
                }
                train_log.update(grad_norm_dict)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                train_step += 1
        
        # average train loss
        train_loss_mean = np.mean(train_loss["total"])
        train_loss_std = np.std(train_loss["total"])
        train_bce_loss_mean = np.mean(train_loss["bce"])
        train_alignment_loss_mean = np.mean(train_loss["alignment"])
        
        # evaluation
        track_selection_module.eval()
        roberta.eval()
        eval_metrics = {
            "loss": {
                "total": [],
                "bce": [],
                "alignment": [],
            },
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        }
        eval_video_ids = []
        with torch.no_grad():
            with tqdm(loader_dict["valid"], desc=f"EPOCH [{epoch + 1} / {configs['train']['n_epochs']}]") as pbar:
                for eval_step, batch in enumerate(pbar):
                    video_id = batch['video_id'][0]
                    expression_id = batch['expression_id'][0]
                    eval_video_ids.append(video_id)
                    
                    # object tokens
                    object_tokens = batch["object_tokens"].to(device)
                    
                    # labels
                    labels = (
                        batch["labels"][configs["train"]["positive_metric"]] > configs["train"]["positive_threshold"]
                    ).float().to(device)
                    
                    # get lang tokens
                    encoded_input = tokenizer.batch_encode_plus(
                        batch['expression'], 
                        padding="longest", 
                        return_tensors="pt"
                    ).to(device)
                    lang_tokens = roberta(**encoded_input)
                    def mean_pooling(model_output, attention_mask):
                        token_embeddings = model_output[0]
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    pos_tokens = mean_pooling(lang_tokens, encoded_input['attention_mask']).unsqueeze(1)
                    lang_tokens = lang_tokens.last_hidden_state
                    neg_tokens = track_selection_module.negative_token.weight.clone().unsqueeze(0).repeat(batch_size, 1, 1)
                    
                    # forward
                    score_logits, score_tokens = track_selection_module(object_tokens, lang_tokens)
                    
                    # binary cross entropy loss
                    weight = torch.ones_like(labels)
                    weight[labels > 0] = configs["train"]["positive_weight"]
                    bce_loss = binary_cross_entropy_with_logits(
                        input=score_logits,
                        target=labels,
                        weight=weight,
                    )
                    # alignment loss
                    alignment_loss = alignment_loss_fn(
                        object_tokens=score_tokens,
                        labels=labels,
                        pos_tokens=pos_tokens,
                        neg_tokens=neg_tokens,
                    )
                    # total loss
                    loss = bce_loss + alignment_loss * configs["train"]["alignment_weight"]
                    
                    # log
                    eval_metrics["loss"]["total"].append(loss.item())
                    eval_metrics["loss"]["bce"].append(bce_loss.item())
                    eval_metrics["loss"]["alignment"].append(alignment_loss.item())
                    
                    # get tp, fp, fn, tn
                    pred_scores = torch.sigmoid(score_logits)
                    preds = (pred_scores > configs["train"]["pred_threshold"]).float()
                    eval_metrics["tp"] += torch.sum((preds == 1) & (labels == 1)).item()
                    eval_metrics["fp"] += torch.sum((preds == 1) & (labels == 0)).item()
                    eval_metrics["fn"] += torch.sum((preds == 0) & (labels == 1)).item()
                    eval_metrics["tn"] += torch.sum((preds == 0) & (labels == 0)).item()
                    
            # average loss
            eval_loss_mean = np.mean(eval_metrics["loss"]["total"])
            eval_loss_std = np.std(eval_metrics["loss"]["total"])
            eval_bce_loss_mean = np.mean(eval_metrics["loss"]["bce"])
            eval_alignment_loss_mean = np.mean(eval_metrics["loss"]["alignment"])
            
            # metrics
            eval_acc = (eval_metrics["tp"] + eval_metrics["tn"]) / (eval_metrics["tp"] + eval_metrics["tn"] + eval_metrics["fp"] + eval_metrics["fn"])
            eval_precision = eval_metrics["tp"] / (eval_metrics["tp"] + eval_metrics["fp"] + 1e-6)
            eval_recall = eval_metrics["tp"] / (eval_metrics["tp"] + eval_metrics["fn"] + 1e-6)
            eval_f1 = 2 * (eval_precision * eval_recall) / (eval_precision + eval_recall + 1e-6)
            
            # log
            with open(os.path.join(configs["results"]["output_dir"], "log.txt"), "a") as f:
                f.write(f"EPOCH {epoch + 1:03d}\n")
                f.write(f"TRAIN EPOCH {epoch + 1:03d} | LOSS: {train_loss_mean:.4f} ({train_loss_std:.4f}) | BCE: {train_bce_loss_mean:.4f} | ALIGNMENT: {train_alignment_loss_mean:.4f}\n")
                f.write(f"VALID EPOCH {epoch + 1:03d} | LOSS: {eval_loss_mean:.4f} ({eval_loss_std:.4f}) | BCE: {eval_bce_loss_mean:.4f} | ALIGNMENT: {eval_alignment_loss_mean:.4f}\n")
                f.write(f"VALID EPOCH {epoch + 1:03d} | ACC: {eval_acc:.4f} | F1: {eval_f1:.4f} | PRECISION: {eval_precision:.4f} | RECALL: {eval_recall:.4f}\n")
                f.write(f"VALID EPOCH {epoch + 1:03d} | TP: {eval_metrics['tp']} | FP: {eval_metrics['fp']} | FN: {eval_metrics['fn']} | TN: {eval_metrics['tn']}\n")
            
            # lr scheduler
            lr_scheduler.step(metrics=eval_loss_mean)
        
        # save model
        torch.save(track_selection_module.state_dict(), os.path.join(configs["results"]["output_dir"], f"epoch_{epoch+1}.pth"))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, unknown = parser.parse_known_args()

    assert args.config is not None, "config file must be provided"
    with open(os.path.join("configs", f"{args.config}.yaml"), "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # UNKNOWN ARGS
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith('--'):
                value = unknown[i+1]
                if value.replace('.', '').isdigit():
                    value = float(value) if '.' in value else int(value)
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                configs[key] = value
                i += 2
            else:
                configs[key] = True
                i += 1
        else:
            i += 1
    
    # SET TRAIN OUTPUT DIR
    configs["results"]["output_dir"] = os.path.join(
        configs["results"]["output_dir"],
        configs["exp_name"],
        configs["dataset"]["train"]['data_name']
    )
    
    print(f"\nTRAINED MODEL WILL BE SAVED IN : {configs['results']['output_dir']}\n")
    os.makedirs(configs["results"]["output_dir"], exist_ok=True)
    
    return configs

if __name__ == "__main__":
    configs = get_configs()
    set_seed(42)
    train(configs)
