<p align="center">
  <h1 align="center">Referring Video Object Segmentation via Language-aligned Track Selection</h1>
  
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-2.6.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg?style=&logo=python&logoColor=ffdd54" alt="Python">
  </p>

  <p align="center">
    <a href="https://github.com/deep-overflow" target="_blank">Seongchan Kim</a>,</span>
    <a href="https://github.com/wooj0216" target="_blank">Woojeong Jin</a>,</span>
    <a href="https://github.com/SangbeomLim" target="_blank">Sangbeom Lim</a>,</span>
    <a href="https://github.com/yoon-heez" target="_blank">Heeji Yoon</a>,</span>
    <a href="https://github.com/Eenrue" target="_blank">Hyunwook Choi</a>,</span>
    <a href="https://cvlab.kaist.ac.kr/members/faculty" target="_blank">Seungryong Kim</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2412.01136">Paper </a> | <a href="https://cvlab-kaist.github.io/SOLA/">Project Page </a> | <a href=#BibTeX>BibTeX </a> </h3>
  <div align="center"></div>

  <p align="center">
    <span style="font-size:18px; color:#d6336c;"><strong>⬇️ Click to watch!</strong></span><br>
    <a href="https://youtu.be/RP_va3dFKFU">
      <img src="https://img.youtube.com/vi/RP_va3dFKFU/0.jpg" alt="Watch the video", width="600px">
    </a>
  </p>

</p>

## Environment Settings

```
conda create -n SOLA python=3.10
conda activate SOLA

pip install -r requirements.txt
```

As our work requires [SAM2](https://github.com/facebookresearch/sam2) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), please follow the **Installation guides** [[`SAM2`](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md), [`GroundingDINO`](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install)] from each repository.

You need to clone each repository in `track_generation` directory.

```
cd track_generation

git clone https://github.com/facebookresearch/sam2.git
cd sam2
(continue with instructions in SAM2 repository)

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GrondingDINO
(continue with instructions in GroundingDINO repository)

cd ../..
```

## Dataset Preparation
You have to download [MeViS](https://codalab.lisn.upsaclay.fr/competitions/15094) and [Ref-Youtube-VOS](https://codalab.lisn.upsaclay.fr/competitions/13520) in `datasets` folder.

The datasets have to be prepared like this:
```
dataset/
    
    mevis/
        train/
            JPEGImages/
                meta_expressions.json
                mask_dict.json
        valid_u/
            ...
        valid/
            ...
    
    ref-ytbvos/
        train/
            Annotations/
            JPEGImages/
            meta_expressions.json
        valid/
            ...
```

## Track Generation

For **Track Generation**, use the code inside the `track_generation` directory, which assumes to include both the `SAM2` and `GroundingDINO` repositories.

Each dataset and split requires both `Prompt Generation` and `Track Generation`.

You can refer to the `scripts` directory for usage examples.

### MeViS train / valid_u / valid
```bash
# MeViS (train) - GT
CUDA_VISIBLE_DEVICES=0 python generate_tokens_GT_mevis.py --dataset mevis --data_type train --pid 0 --n_pids 1

# MeViS (train) - GRID
CUDA_VISIBLE_DEVICES=0 python generate_prompts_grid.py --dataset mevis --data_type train --bin_size 4 --pid 0 --n_pid 1
CUDA_VISIBLE_DEVICES=0 python generate_tokens_grid.py --dataset mevis --data_type train --bin_size 4 --batch_size 4 --miou_thresh 0.7 --n_max_tracks 64 --pid 0 --n_pids 1

# MeViS (valid_u / valid) - GroundingDINO
CUDA_VISIBLE_DEVICES=0 python generate_prompts_gdino.py --dataset mevis --data_type valid_u --bin_size 4 --box_threshold 0.2 --text_threshold 0.25 --pid 0 --n_pid 1
CUDA_VISIBLE_DEVICES=0 python generate_tokens_gdino.py --dataset mevis --data_type valid_u --bin_size 4 --batch_size 4 --miou_thresh 0.7 --stability_score_thresh 0.85 --n_max_tracks 16 --pid 0 --n_pids 1

# MeViS (valid_u / valid) - GRID
CUDA_VISIBLE_DEVICES=0 python generate_prompts_grid.py --dataset mevis --data_type valid_u --bin_size 0 --pid 0 --n_pid 1
CUDA_VISIBLE_DEVICES=0 python generate_tokens_grid.py --dataset mevis --data_type valid_u --bin_size 4 --batch_size 4 --miou_thresh 0.7 --n_max_tracks 64 --pid 0 --n_pids 1
```

### Ref-Youtube-VOS train / valid
```bash
# Ref-Youtube-VOS (train) - GT
CUDA_VISIBLE_DEVICES=0 python generate_tokens_GT_ytbvos.py --dataset ref-ytbvos --data_type train --pid 0 --n_pid 1

# Ref-Youtube-VOS (train) - GRID
CUDA_VISIBLE_DEVICES=0 python generate_prompts_grid.py --dataset ref-ytbvos --data_type train --bin_size 4 --pid 0 --n_pid 1
CUDA_VISIBLE_DEVICES=0 python generate_tokens_grid.py --dataset ref-ytbvos --data_type train --bin_size 4 --batch_size 4 --miou_thresh 0.7 --n_max_tracks 64 --pid 0 --n_pids 1

# Ref-Youtube-VOS (valid) - GroundingDINO
CUDA_VISIBLE_DEVICES=0 python generate_prompts_gdino.py --dataset ref-ytbvos --data_type valid --bin_size 4 --box_threshold 0.2 --text_threshold 0.25 --pid 0 --n_pid 1
CUDA_VISIBLE_DEVICES=0 python generate_tokens_gdino.py --dataset ref-ytbvos --data_type valid --bin_size 4 --batch_size 4 --miou_thresh 0.7 --stability_score_thresh 0.85 --n_max_tracks 16 --pid 0 --n_pids 1

# Ref-Youtube-VOS (valid) - GRID
CUDA_VISIBLE_DEVICES=0 python generate_prompts_grid.py --dataset ref-ytbvos --data_type valid --bin_size 0 --pid 0 --n_pid 1
CUDA_VISIBLE_DEVICES=0 python generate_tokens_grid.py --dataset ref-ytbvos --data_type valid --bin_size 4 --batch_size 4 --miou_thresh 0.7 --n_max_tracks 64 --pid 0 --n_pids 1
```

These will generate SAM2 object tokens and corresponding masklets in `sam2_tracks` directory.


## Track Selection
After generating SAM2 tracks and object tokens, you have to train and inference to obatain the final results.

You can use the `scripts` directory for simple usage.

```bash
# Training
sh train.sh mevis/default

# Evaluation
sh eval.sh mevis/default [epoch] --eval_pred_threshold [threshold]

# Inference
sh inference.sh mevis/default [epoch] --eval_pred_threshold [threshold]
```

To obtain Zero-shot results:
```bash
# Zero-shot Evaluation
sh eval.sh mevis/zeroshot [epoch] --eval_pred_threshold [threshold]

# Zero-shot Inference
sh inference.sh mevis/zeroshot [epoch] --eval_pred_threshold [threshold]
```

<!-- ## Models

☁️ [Google Drive](???)

## Acknowledgement

This project is based on ???. Many thanks to the authors for their great works! -->

## BibTeX
<p id="BibTeX"></p>

Please consider to cite **SOLA** if it helps your research.

```bibtex
@article{kim2024referring,
  title={Referring Video Object Segmentation via Language-aligned Track Selection},
  author={Kim, Seongchan and Jin, Woojeong and Lim, Sangbeom and Yoon, Heeji and Choi, Hyunwook and Kim, Seungryong},
  journal={arXiv preprint arXiv:2412.01136},
  year={2024}
}
```
