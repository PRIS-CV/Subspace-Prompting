# custom config
DATA=/mnt/sdb/data/datasets
TRAINER=SuPr
WEIGHTSPATH=weights

DATASET=$1 # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
CFG=vit_b16_ep25_batch8_4+4ctx_few_shot
for SEED in 1 2 3
do
    TRAIL_NAME=reproduce_${CFG}
    COMMON_DIR=${DATASET}/shots_4/seed${SEED}
    MODEL_DIR=${WEIGHTSPATH}/${TRAINER}/fewshot/${COMMON_DIR}
    DIR=output/fewshot/${TRAINER}/${TRAIL_NAME}/${COMMON_DIR}

    if [ -d "$DIR" ]; then
        echo " The results exist at ${DIR}"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --eval-only 
    fi
done

