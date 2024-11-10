#!/usr/bin/env bash

CONFIG='configs/distillers/advanced_training_strategy/swin-l_distill_mlp_s_img_s3_s4.py'

echo $MLP_ROLE_INDEX
GPU=8

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --node_rank=$MLP_ROLE_INDEX \
    --master_port=$MLP_WORKER_0_PORT \
    --nnodes=$MLP_WORKER_NUM\
    tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}\
