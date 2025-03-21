import os
import yaml
import torch
import argparse
import numpy as np
import random

from module.module import LanguageAlignedTrackSelectionModule
from evaluator import Evaluator


def eval(configs):
    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MODEL
    model = LanguageAlignedTrackSelectionModule(configs["model"]).to(device)
    
    # LOAD WEIGHTS
    eval_weights_epoch = configs["eval_weight_epoch"]
    weights_path = os.path.join(
        configs["results"]["output_dir"], 
        configs["dataset"]["train"]["data_name"], 
        f"epoch_{eval_weights_epoch}.pth"
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    print(f"loaded weights from {weights_path}")

    # EVALUATOR
    evaluator = Evaluator(
        model=model,
        configs=configs,
        eval_weight_epoch=eval_weights_epoch,
        device=device,
    )
    evaluator.evaluate()

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
    
    # SET EVAL OUTPUT DIR
    configs["results"]["output_dir"] = os.path.join(configs["results"]["output_dir"], configs["exp_name"])
    configs["results"]["eval_output_dir"] = os.path.join(
        configs["results"]["eval_output_dir"],
        configs["exp_name"],
        configs["dataset"]["valid"]["data_name"],
        f"pred_threshold_{str(configs['eval']['pred_threshold']).replace('.', '')}",
        f"epoch_{configs['eval_weight_epoch']}",
    )
    
    print("EVAL OUTPUTS WILL BE SAVED IN", configs["results"]["eval_output_dir"])
    
    os.makedirs(configs["results"]["eval_output_dir"], exist_ok=True)
    
    return configs

if __name__ == "__main__":
    configs = get_configs()
    set_seed(42)
    eval(configs)
