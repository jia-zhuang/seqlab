#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM="true"

export NPROC=1

export ROOT=.

export TASK=multihead_labeling
export MAX_LENGTH=100
export BERT_MODEL=${ROOT}/assets/models/bert-base-chinese/
export OUTPUT_DIR=${ROOT}/assets/models/multihead_sequence_labeling/
export DATA_DIR=${ROOT}/assets/data/multihead_sequence_labeling/
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SEED=42
export IS_SMALL=False

# python -m torch.distributed.launch --nproc_per_node=$NPROC --nnodes=1 train.py \
python train.py \
--task $TASK \
--data_dir $DATA_DIR \
--is_small $IS_SMALL \
--labels labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--warmup_ratio 0.1 \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--evaluation_strategy epoch \
--logging_strategy epoch \
--save_strategy epoch \
--report_to tensorboard \
--dataloader_num_workers 2 \
--seed $SEED \
--do_train \
--do_eval

