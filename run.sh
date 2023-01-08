#!/usr/bin/env bash

datasets=("mutag" "alkane" "DD" "PROTEIN")
architectures=("gcn" "graphconv" "leconv")
explainers=("gnn_explainer" "pgmexplainer" "intgradexplainer")
now=$(date +%F-%R)
logdir="logs/${now}.log"
mkdir -p logs

for dataset in ${datasets[@]}; do
    for architecture in ${architectures[@]}; do
        for explainer in ${explainers[@]}; do

            printf "\n\n[dataset: '%12s'][architecture: '%12s'][explainer: '%12s'][vanilla: 'OFF']\n" \
                $dataset $architecture $explainer | tee -a ${logdir}
            python train.py \
                --epochs 27 \
                --explainer_epochs 25 \
                --dataset $dataset \
                --arch $architecture \
                --explainer $explainer 2>&1 | tee -a ${logdir}

            printf "\n\n[dataset: '%12s'][architecture: '%12s'][explainer: '%12s'][vanilla: 'ON']\n" \
                $dataset $architecture $explainer | tee -a ${logdir}
            python train.py \
                --epochs 27 \
                --explainer_epochs 25 \
                --vanilla_mode \
                --dataset $dataset \
                --arch $architecture \
                --explainer $explainer 2>&1 | tee -a ${logdir} 
        done
    done
done