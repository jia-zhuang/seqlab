#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

export NPROC=1

export ROOT=.

export TASK=multi_head_labeling
export MAX_LENGTH=200
export BERT_MODEL=${ROOT}/models/debug1/
export OUTPUT_DIR=${ROOT}/models/debug1/
export DATA_DIR=${ROOT}/datasets/raw/
export BATCH_SIZE=24
export NUM_EPOCHS=2
export SAVE_STEPS=500000
export LOGGING_STEPS=500
export SEED=42

python -m torch.distributed.launch --nproc_per_node=$NPROC --nnodes=1 train.py \
--task $TASK \
--data_dir $DATA_DIR \
--labels labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--logging_steps $LOGGING_STEPS \
--seed $SEED \
--do_eval

# --do_train \
