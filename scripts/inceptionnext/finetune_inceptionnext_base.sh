DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/inceptionnext # modify code path here
INIT_CKPT=/path/to/trained/model.pth


ALL_BATCH_SIZE=1024
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


MODEL=inceptionnext_base_384
DROP_PATH=0.7
DROP=0.5


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --img-size 384 --epochs 30 --opt adamw --lr 5e-5 --sched None \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--initial-checkpoint $INIT_CKPT \
--mixup 0 --cutmix 0 \
--model-ema --model-ema-decay 0.9999 \
--drop-path $DROP_PATH --drop $DROP