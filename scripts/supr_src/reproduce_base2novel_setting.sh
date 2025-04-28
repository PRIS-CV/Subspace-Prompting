# custom config
DATA=/mnt/sdb/data/datasets
TRAINER=SubspacePromptSRC
WEIGHTSPATH=weights


DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'

EP=20
CFG=vit_b16_ep20_batch4_4+4ctx_promptsrc
SHOTS=16

for SEED in 1 2 3
do  
    TRAIL_NAME=reproduce_${CFG}
    COMMON_DIR=${DATASET}/shots_${SHOTS}/seed${SEED}
    MODEL_DIR=${WEIGHTSPATH}/${TRAINER}/${COMMON_DIR}
    BASE_DIR=output/base2new/${TRAINER}/${TRAIL_NAME}/train_base/${COMMON_DIR}
    NEW_DIR=output/base2new/${TRAINER}/${TRAIL_NAME}/test_new/${COMMON_DIR}

    if [ -d "$NEW_DIR" ]; then
        echo " The results exist at ${NEW_DIR}"
    else
        echo "Run this job and save the output to ${NEW_DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/SuPr/${CFG}.yaml \
        --output-dir ${BASE_DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${EP} \
        --eval-only \
        DATASET.SUBSAMPLE_CLASSES base    
    fi  

    if [ -d "$NEW_DIR" ]; then
        echo " The results exist at ${NEW_DIR}"
    else
        echo "Run this job and save the output to ${NEW_DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/SuPr/${CFG}.yaml \
        --output-dir ${NEW_DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${EP} \
        --eval-only \
        DATASET.SUBSAMPLE_CLASSES new
    fi
done
