

# custom config
DATA=/mnt/sdb/data/datasets
TRAINER=SuPr


SHOTS=16
CFG=vit_b16_ep12_batch8_4+4ctx_cross_datasets

for SEED in 1 2 3
do
    TRAIL_NAME=${CFG}
    MODEL_DIR=output/cross_dg/${TRAINER}/${TRAIL_NAME}/imagenet/shots_${SHOTS}/seed${SEED}

    if [ -d "$MODEL_DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/imagenet.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${MODEL_DIR} \
        DATASET.NUM_SHOTS ${SHOTS} 
    fi

    # cross
    for DATASET in caltech101 dtd eurosat fgvc_aircraft oxford_flowers food101 oxford_pets stanford_cars sun397 ucf101
    do
        DIR=output/cross_dg/${TRAINER}/${TRAIL_NAME}/${DATASET}/shots_${SHOTS}/seed${SEED}

        if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            echo "Cross-dataset Evaluating"
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

    # dg
    for DATASET in imagenetv2 imagenet_sketch imagenet_a imagenet_r
    do
        DIR=output/cross_dg/${TRAINER}/${TRAIL_NAME}/${DATASET}/shots_${SHOTS}/seed${SEED}

        if [ -d "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            echo "Domain Generlization Evaluating"
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

done
