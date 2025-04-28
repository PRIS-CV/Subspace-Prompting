#!/bin/bash

#cd ../..

# custom config
DATA=/mnt/sdb/data/datasets
TRAINER=SuPr

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'


SHOTS=4
CFG=vit_b16_ep25_batch8_4+4ctx_few_shot
for SEED in 1 2 3
do
    # COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/1119_s${SUBDIM}_${CFG}/seed${SEED}
    TRAIL_NAME=${CFG}
    DIR=output/fewshot/${TRAINER}/${TRAIL_NAME}/${DATASET}/shots_${SHOTS}/seed${SEED}

    if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}                 
    fi         
done