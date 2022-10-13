#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gvci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name graph-inference-L008"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --gpu 1" #PYARGS="$PYARGS --cpu"

PYARGS="$PYARGS --data $DATA/datasets/L008_prepped.h5ad" #marson_prepped.h5ad, sciplex_prepped.h5ad
PYARGS="$PYARGS --perturbation_key target_gene" #target, perturbation
PYARGS="$PYARGS --covariate_keys cell_type donor" #celltype donor stim, cell_type replicate
PYARGS="$PYARGS --split_key split"
#PYARGS="$PYARGS --dose_key dose"
PYARGS="$PYARGS --graph_path $DATA/graphs/L008_grn_8.pth"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --outcome_dist normal"
PYARGS="$PYARGS --checkpoint_freq 10"

python src/main.py $PYARGS
