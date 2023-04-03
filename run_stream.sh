#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="true"

export NPROC=1

export ROOT=.

export TASK=multihead_labeling
export MAX_LENGTH=20
export BERT_MODEL=${ROOT}/../models/damo_bert.slim/
export OUTPUT_DIR=${ROOT}/../models/resample3/
export DATA_DIR=${ROOT}/../datasets2/resample/
export BATCH_SIZE=2048
export NUM_EPOCHS=6
export MAX_STEPS=20577    # 6 * 7_023_833 / 2048
export LOG_STEPS=343      # 7_023_833 / 2048 / 10   10 point each epoch
export SEED=42
export IS_SMALL=False

# python -m torch.distributed.launch --nproc_per_node=$NPROC --nnodes=1 train.py \
python3  train.py \
--task $TASK \
--data_dir $DATA_DIR \
--is_small $IS_SMALL \
--labels entity_dim.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--max_steps $MAX_STEPS \
--learning_rate 5e-5 \
--warmup_ratio 0.1 \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--evaluation_strategy steps \
--save_strategy steps \
--logging_strategy steps \
--logging_steps $LOG_STEPS \
--save_steps $LOG_STEPS \
--report_to tensorboard \
--dataloader_num_workers 8 \
--dataloader_pin_memory \
--seed $SEED \
--do_train \
--do_eval
