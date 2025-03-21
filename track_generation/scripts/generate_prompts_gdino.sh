CUDA_VISIBLE_DEVICES=0 python generate_prompts_gdino.py \
    --dataset mevis \
    --data_type valid \
    --bin_size 4 \
    --box_threshold 0.2 \
    --text_threshold 0.25 \
    --pid 0 \
    --n_pid 1
