GPUS=3,1,2,0
export CUDA_VISIBLE_DEVICES=$GPUS

#BASE_ROOT=/root/Workspace
IMAGE_DIR=$BASE_ROOT/datasets
ANNO_DIR=$BASE_ROOT/datasets/processed_data

CKPT_DIR=$BASE_ROOT/modelZoo
LOG_DIR=$BASE_ROOT/data/resnet50/logs_90
PRETRAINED_PATH=$BASE_ROOT/pretrained/resnet50-19c8e357.pth
IMAGE_MODEL=resnet50
EMBEDDING_INIT_PATH=$BASE_ROOT/datasets/processed_data/text_embedding.pkl

lr=0.00032

num_epoches=300
batch_size=128
lr_decay_ratio=0.9
epoches_decay=20_30_40

python ${BASE_ROOT}/test.py \
    --bidirectional \
    --model_path $CKPT_DIR \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --gpus $GPUS \
    --epoch_start 0 \
    --checkpoint_dir $CKPT_DIR \
    --embedding_init_path $EMBEDDING_INIT_PATH \
#    --re_ranking yes
#    --wd 0.0004


