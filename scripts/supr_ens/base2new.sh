

# custom config
DATA=/mnt/sdb/data/datasets
TRAINER=SuPrEns

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'

SHOTS=16
EP=10
CFG=vit_b16_ep10_batch4_4+4ctx

for SEED in 1 2 3
do
    TRAIL_NAME=${CFG}
    COMMON_DIR=${DATASET}/shots_${SHOTS}/seed${SEED}
    MODEL_DIR=output/base2new/${TRAINER}/${TRAIL_NAME}/train_base/${COMMON_DIR}
    DIR=output/base2new/${TRAINER}/${TRAIL_NAME}/test_new/${COMMON_DIR}

    if [ -d "$MODEL_DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/SuPr/${CFG}.yaml \
        --output-dir ${MODEL_DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.SUPR.SVD False \
        DATASET.SUBSAMPLE_CLASSES base    
        # use least square save gpu memory(mathematical equivalent)
    fi
    if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "Evaluating model"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/SuPr/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${EP} \
        --eval-only \
        TRAINER.SUPR.SVD False \
        DATASET.SUBSAMPLE_CLASSES new
        # use least square save gpu memory(mathematical equivalent)
    fi
done
