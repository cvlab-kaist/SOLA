import os
import yaml
import torch
import argparse
import numpy as np
import random
from tqdm import tqdm
import imageio.v2 as iio

from transformers import AutoTokenizer, AutoModel

from module.module import LanguageAlignedTrackSelectionModule
from dataloader import get_loader_dict

@torch.no_grad()
def inference(configs):
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MODEL
    model = LanguageAlignedTrackSelectionModule(configs["model"]).to(device)
    roberta = AutoModel.from_pretrained(configs["model"]["roberta_version"]).to(device)
    roberta.eval()
    tokenizer = AutoTokenizer.from_pretrained(configs["model"]["roberta_version"])
    
    # LOAD WEIGHTS
    eval_weights_epoch = configs["eval_weight_epoch"]
    weights_path = os.path.join(
        configs["results"]["output_dir"], 
        configs["dataset"]["train"]["data_name"], 
        f"epoch_{eval_weights_epoch}.pth"
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    print(f"loaded weights from {weights_path}")
    model.eval()

    # DATALOADERS
    loader_dict = get_loader_dict(
        configs=configs["dataset"],
        only_inference=True,
    )
    # INFERENCE
    pred_dict = {}
    with tqdm(loader_dict["test"], desc="INFERENCE", ascii="░▒▓█", colour="CYAN") as pbar:
        for batch in pbar:
            # object tokens
            object_tokens = batch["object_tokens"].to(device)

            # get lang tokens
            lang_tokens = tokenizer.batch_encode_plus(
                batch["expression"],
                padding="longest",
                return_tensors="pt",
            ).to(device)
            lang_tokens = roberta(**lang_tokens).last_hidden_state
            
            # forward
            pred_score, _ = model(object_tokens, lang_tokens)
            pred_score = torch.sigmoid(pred_score)
            pred = (pred_score > configs["eval"]["pred_threshold"]).float()
            
            # pred dict
            video_id = batch["video_id"][0]
            expression_id = batch["expression_id"][0]
            pred_dict[video_id] = pred_dict.get(video_id, {})
            pred_dict[video_id][expression_id] = {
                "expression": batch["expression"][0],
                "pred": pred.cpu().numpy()[0],
                "pred_score": pred_score.cpu().numpy()[0],
                "root_type": batch["root_type"][0],
                "prompt_type": batch["prompt_type"][0],
                "sam2_anno_id": batch["sam2_anno_id"][0],
            }
    
    # SAVE PREDICTIONS
    for video_idx, video_id in enumerate(pred_dict):
        frames = loader_dict["test"].dataset.get_frames(video_id)
        with tqdm(pred_dict[video_id].items(), desc=f"SAVE PREDICTIONS {video_id} [{video_idx+1}/{len(pred_dict)}]", ascii="░▒▓█", colour="CYAN") as pbar:
            for expression_id, pred_info in pbar:
                pred_masklet = loader_dict["test"].dataset.get_sam2_masklet(
                    video_id=video_id,
                    expression_id=expression_id,
                    preds=pred_info["pred"],
                    root_types=pred_info["root_type"],
                    prompt_types=pred_info["prompt_type"],
                    sam2_anno_ids=pred_info["sam2_anno_id"],
                )
                os.makedirs(os.path.join(configs["results"]["test_output_dir"], video_id, expression_id), exist_ok=True)
                assert pred_masklet is not None, f"pred_masklet is None for {video_id}/{expression_id}"
                for frame_id, mask in zip(frames, pred_masklet):
                    mask = (mask * 255).astype(np.uint8)
                    iio.imwrite(os.path.join(configs["results"]["test_output_dir"], video_id, expression_id, f"{frame_id}.png"), mask)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--eval_weight_epoch", type=int, default=-1)
    parser.add_argument("--eval_pred_threshold", type=float, default=0.5)
    args, unknown = parser.parse_known_args()
    
    assert args.config is not None, "config file must be provided"
    with open(os.path.join("configs", f"{args.config}.yaml"), "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    configs["eval"]["pred_threshold"] = args.eval_pred_threshold
    
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
    
    # SET EVAL WEIGHT EPOCH
    configs["eval_weight_epoch"] = args.eval_weight_epoch
    
    # SET TEST OUTPUT DIR
    configs["results"]["output_dir"] = os.path.join(configs["results"]["output_dir"], configs["exp_name"])
    configs["results"]["test_output_dir"] = os.path.join(
        configs["results"]["test_output_dir"],
        configs["exp_name"],
        configs["dataset"]["test"]["data_name"],
        f"pred_threshold_{str(configs['eval']['pred_threshold']).replace('.', '')}",
        f"epoch_{configs['eval_weight_epoch']}",
    )
    
    print("INFERENCE OUTPUTS WILL BE SAVED IN", configs["results"]["test_output_dir"])
    
    return configs

if __name__ == "__main__":
    configs = get_configs()
    set_seed(42)
    inference(configs)