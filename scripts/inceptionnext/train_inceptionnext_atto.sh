DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/inceptionnext # modify code path here


ALL_BATCH_SIZE=1280
NUM_GPU=4
GRAD_ACCUM_STEPS=1
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


MODEL=inceptionnext_atto
DROP_PATH=0.1


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL_NAME --opt adamw --lr 1e-3 --warmup-epochs 5 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DATA_PATH \
--aa rand-m5-inc1-mstd101 \
--color-jitter 0 \
--epochs 600 \
--mixup 0.2 \
--reprob 0.1 \
--warmup-lr 5e-7 --min-lr 5e-7 \
--no-cj