#!/bin/bash

# BASE ARGUMENTS
CUDA_DEVICE=$1
CONFIG=$2
EVAL_WEIGHT_EPOCH=$3

# EXTRA ARGUMENTS
EXTRA_ARGS=""

shift 3
while [ $# -gt 0 ]; do
    EXTRA_ARGS="$EXTRA_ARGS $1 $2"
    shift 2
done

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python inference.py --config $CONFIG --eval_weight_epoch $EVAL_WEIGHT_EPOCH $EXTRA_ARGS