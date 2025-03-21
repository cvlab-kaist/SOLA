CUDA_VISIBLE_DEVICES=0 python generate_tokens_grid.py \
    --dataset mevis \
    --data_type train \
    --bin_size 4 \
    --batch_size 4 \
    --miou_thresh 0.7 \
    --n_max_tracks 64 \
    --pid 0 \
    --n_pids 1
