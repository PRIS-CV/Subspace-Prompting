#!/bin/bash
# custom config
DATA=/mnt/sdb/data/datasets
TRAINER=SuPr
WEIGHTSPATH=weights

DATASET=$1 
CFG=vit_b16_ep12_batch8_4+4ctx_cross_datasets


for SEED in 1 2 3
do  
    TRAIL_NAME=reproduce_${CFG}
    MODEL_DIR=${WEIGHTSPATH}/${TRAINER}/cross_dg/imagenet/shots_16/seed${SEED}
    DIR=output/cross_dg/${TRAINER}/${TRAIL_NAME}/${DATASET}/shots_16/seed${SEED}
    if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "Evaluating"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch 4 \
        --eval-only 
    fi
done


