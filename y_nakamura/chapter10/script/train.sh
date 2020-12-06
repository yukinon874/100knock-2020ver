### training script ###
. setting.sh

mkdir -p ${MODEL_DIR}
mkdir -p ${LOG_DIR}

CUDA_VISIBLE_DEVICES=${GPU} fairseq-train ${DATA_DIR} \
    -s ja -t en \
    --arch transformer_iwslt_de_en \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --update-freq 64 \
    --max-update 5000 \
    --fp16 \
    --keep-last-epochs 5 \
    --tensorboard-logdir ${LOG_DIR} \
    --log-format simple \
    --seed ${SEED} | tee -a ${LOG_DIR}/train.log
