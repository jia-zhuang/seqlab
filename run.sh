#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1,2"

export NPROC=2

export ROOT=/workspace/workspace/spo2020/relation_multi_label

export MAX_LENGTH=200
export BERT_MODEL=${ROOT}/models/roberta-base-chinese/
export OUTPUT_DIR=${ROOT}/models/test3/
export DATA_DIR=${ROOT}/datasets/
export BATCH_SIZE=24
export NUM_EPOCHS=2
export SAVE_STEPS=500000
export LOGGING_STEPS=500
export SEED=42

python -m torch.distributed.launch --nproc_per_node=$NPROC --nnodes=1 run_multi_label.py \
--data_dir $DATA_DIR \
--labels labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--logging_steps $LOGGING_STEPS \
--seed $SEED \
--do_train \
--do_eval

# --task_name bert \