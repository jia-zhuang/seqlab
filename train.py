import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch import nn

from transformers import (
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers import BertForTokenClassification
from tokenizers import BertWordPieceTokenizer

from utils import SeqLabelDataset, get_labels

from multi_head import BertForMultiHeadTokenClassification


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    task: Optional[str] = field(
        default='labeling',
        metadata={"help": "Task, can be `labeling` or `multi_head_labeling`"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = get_labels(data_args.labels)
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = BertWordPieceTokenizer(os.path.join(model_args.model_name_or_path, 'vocab.txt'))
    
    model_cls = BertForMultiHeadTokenClassification if model_args.task == 'multi_head_labeling' else BertForTokenClassification
    model = model_cls.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)

    # Get datasets
    train_dataset = (
        SeqLabelDataset(
            task=model_args.task,
            data_dir=data_args.data_dir,
            mode='train',
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        SeqLabelDataset(
            task=model_args.task,
            data_dir=data_args.data_dir,
            mode='dev',
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        if model_args.task == 'labeling':
            # predictions: N x L x S
            # label_ids: N x L
            label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
            preds = np.argmax(predictions, axis=2)
        elif model_args.task == 'multi_head_labeling':
            # predictions: N x num_heads x L x S
            # label_ids: N x num_heads x L
            preds = np.argmax(predictions, axis=3)   # N x num_heads x L
            N, num_heads, seq_len = preds.shape

            label_map = {0: 'O'}
            for i, label in enumerate(labels):
                label_map[2*i + 1] = f'B-{label}'
                label_map[2*i + 2] = f'I-{label}'

            def expand_heads(ids):
                ''' ids: label_ids or preds, shape=(N, num_heads, L)
                    outputs: shape=(N, num_heads*L)
                '''
                outputs = []
                for i, _ in enumerate(labels):
                    head_ids = ids[:, i]
                    outputs.append(
                        np.where((head_ids==1)|(head_ids==2), head_ids + 2*i, head_ids)
                    )

                outputs = np.concatenate(outputs, axis=1)  # N x num_heads*L
                return outputs
            
            preds = expand_heads(preds)
            label_ids = expand_heads(label_ids)
        
        else:
            raise ValueError(f'Error! Invalid task: `f{model_args.task}`')

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        report_str = classification_report(out_label_list, preds_list)
        print(report_str)
        # save report
        with open(os.path.join(training_args.output_dir, 'report.txt'), 'w') as f:
            f.write(report_str)

        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
    
    def data_collator(features) -> Dict[str, torch.Tensor]:
        batch = {}
        for k in ('input_ids', 'attention_mask', 'token_type_ids'):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        
        batch['labels'] = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        return batch

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        if trainer.is_world_master():
            tokenizer.save_model(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict and training_args.local_rank in [-1, 0]:
        test_dataset = NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            local_rank=training_args.local_rank,
        )

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, _ = align_predictions(predictions, label_ids)

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not preds_list[example_id]:
                            example_id += 1
                    elif preds_list[example_id]:
                        output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    return results


if __name__ == "__main__":
    main()
