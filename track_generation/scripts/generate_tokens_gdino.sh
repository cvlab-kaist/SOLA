CUDA_VISIBLE_DEVICES=0 python generate_tokens_gdino.py \
    --dataset mevis \
    --data_type valid \
    --bin_size 4 \
    --batch_size 4 \
    --miou_thresh 0.7 \
    --stability_score_thresh 0.85 \
    --n_max_tracks 16 \
    --pid 0 \
    --n_pids 1
