#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gvci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name graph-inference-marson"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --gpu 1" #PYARGS="$PYARGS --cpu"

PYARGS="$PYARGS --data $DATA/datasets/marson_prepped.h5ad" #sciplex_prepped.h5ad, L008_prepped.h5ad
PYARGS="$PYARGS --perturbation_key target" #perturbation, target_gene
PYARGS="$PYARGS --covariate_keys celltype donor stim" #cell_type replicate, cell_type donor
PYARGS="$PYARGS --split_key split"
#PYARGS="$PYARGS --dose_key dose"
PYARGS="$PYARGS --graph_path $DATA/graphs/marson_grn_8.pth"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --outcome_dist normal"
PYARGS="$PYARGS --checkpoint_freq 10"

python src/main.py $PYARGS
