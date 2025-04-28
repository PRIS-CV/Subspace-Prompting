# custom config
DATA=/mnt/sdb/data/datasets
TRAINER=SuPrEns
WEIGHTSPATH=weights


DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
EP=10
CFG=vit_b16_ep10_batch4_4+4ctx
SHOTS=16

for SEED in 1 2 3
do  
    TRAIL_NAME=reproduce_${CFG}
    COMMON_DIR=${DATASET}/shots_${SHOTS}/seed${SEED}
    MODEL_DIR=${WEIGHTSPATH}/${TRAINER}/${COMMON_DIR}
    BASE_DIR=output/base2new/${TRAINER}/${TRAIL_NAME}/train_base/${COMMON_DIR}
    NEW_DIR=output/base2new/${TRAINER}/${TRAIL_NAME}/test_new/${COMMON_DIR}

    
    if [ -d "$BASE_DIR" ]; then
        echo " The results exist at ${BASE_DIR}"
    else
        echo "Run this job and save the output to ${BASE_DIR}"
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
        TRAINER.SUPR.SVD False \
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
        TRAINER.SUPR.SVD False \
        DATASET.SUBSAMPLE_CLASSES new
    fi
done
