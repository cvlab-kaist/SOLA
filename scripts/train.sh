#!/bin/bash

# BASE ARGUMENTS
CUDA_DEVICE=$1
CONFIG=$2

# EXTRA ARGUMENTS
EXTRA_ARGS=""

shift 2
while [ $# -gt 0 ]; do
    EXTRA_ARGS="$EXTRA_ARGS $1"
    shift
done

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --config $CONFIG $EXTRA_ARGS