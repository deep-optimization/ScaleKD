#!/usr/bin/env bash

CONFIG=$1
GPU=$2
MLP_ROLE_INDEX=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --node_rank=$MLP_ROLE_INDEX \
    --master_port=$MLP_WORKER_0_PORT \
    --nnodes=4\
    tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}\
