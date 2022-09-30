#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gvci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name pretrain_epoch-100"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --gpu 0" #PYARGS="$PYARGS --cpu"

PYARGS="$PYARGS --graph_path $DATA/graphs/marson_grn.pth"

PYARGS="$PYARGS --hparams link_hparams.json"

PYARGS="$PYARGS --max_epochs 100"
PYARGS="$PYARGS --batch_size 128"
PYARGS="$PYARGS --checkpoint_freq 5"

python link_pred.py $PYARGS
