import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from filelock import FileLock

from auto_tokenize_utils import TokenizedSentence


logger = logging.getLogger(__name__)


''' Example schema
class Example:
    sentence: str
    prefix: Optional[str]
    labels: Dict[span, tag]  # {(2, 4): PER, (5, 10): LOC}
'''


@dataclass
class InputFeature:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Union[List[int], List[List[int]], None] = None    # for labeling, multihead_labeling and predict


class SeqLabelDataset(Dataset):
    def __init__(self, task, data_dir, mode, is_small, tokenizer, labels, max_seq_length, overwrite_cache=False):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length)),
        )
        if is_small: cached_features_file += '_small'

        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                # read examples
                data_file = f'{mode}_small.pkl' if is_small else f'{mode}.pkl'
                examples = pd.read_pickle(os.path.join(data_dir, data_file))
                # convert examples to features
                TokenizedSentence.setup(tokenizer, max_seq_length)
                label_map = {label: i for i, label in enumerate(labels)}
                
                self.features = []
                for ex_idx, example in enumerate(examples.itertuples()):
                    if ex_idx % 10000 == 0:
                        logger.info("Writing example %d of %d", ex_idx, examples.shape[0])

                    self.features.append(
                        convert_example_to_feature(ex_idx, example, label_map, task)
                    )

                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeature:
        return self.features[i]


def generate_label_ids(token_spans, ignore_mask, ignore_index=-100):
    label_ids = [0] * len(ignore_mask)
    for span, code in token_spans.items():
        if span[0] < span[1]:  # valid span
            label_ids[span[0]] = code  # B
            label_ids[(span[0] + 1): span[1]] = [code + 1] * (span[1] - span[0] - 1)  # I
    
    # ignore mask
    for i, flag in enumerate(ignore_mask):
        if flag == 0:
            label_ids[i] = ignore_index
    
    return label_ids


def convert_example_to_feature(ex_idx, example, label_map, task):
    ''' 先不考虑span重叠的情况
    '''
    prefix = getattr(example, 'prefix', None)
    tokend_sent = TokenizedSentence(example.sentence, prefix=prefix)

    if prefix is None:
        ignore_mask = tokend_sent.attention_mask  # 无prefix(单句)的情况，attention_mask==0正好时需要被ignore的token
    else:
        ignore_mask = tokend_sent.token_type_ids  # 有prefix(上下句)的情况，token_type_ids==0正好时需要被ignore的token

    # 生成 label_ids
    if task == 'labeling':
        token_spans = {}
        for span, tag in example.labels.items():
            token_spans[tokend_sent.char_span_to_token_span(span)] = label_map[f'B-{tag}']  # 这里只用B-tag标识，I-tag通过+1获得，这就要求在写labels.txt文件时，遵循相邻和有序原则
        label_ids = generate_label_ids(token_spans, ignore_mask)

    elif task == 'multihead_labeling':
        label_ids = []
        for head in label_map:  # 要求是 order dict，对于python>=3.6，dict默认是有序的
            token_spans = {}
            for span, tag in example.labels.items():
                if tag == head:
                    token_spans[tokend_sent.char_span_to_token_span(span)] = 1   # {O: 0, B: 1, I: 2}
            
            label_ids.append(
                generate_label_ids(token_spans, ignore_mask)
            )
    else:
        raise ValueError(f'Error! Invalid task: `f{task}`')

    if ex_idx < 5:
        logger.info("*** Example ***")
        logger.info("ex_idx: %s", ex_idx)
        logger.info("tokens: %s", ' '.join(map(str, tokend_sent.tokens)))
        logger.info("input_ids: %s", ' '.join(map(str, tokend_sent.input_ids)))
        logger.info("attention_mask: %s", ' '.join(map(str, tokend_sent.attention_mask)))
        logger.info("token_type_ids: %s", ' '.join(map(str, tokend_sent.token_type_ids)))
        logger.info("label_ids: %s", ' '.join(map(str, label_ids)))

    return InputFeature(
        input_ids=tokend_sent.input_ids,
        attention_mask=tokend_sent.attention_mask,
        token_type_ids=tokend_sent.token_type_ids,
        label_ids=label_ids,
    )


def get_labels(path: str) -> List[str]:
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


if __name__ == "__main__":
    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer('assets/bert-base-chinese-vocab.txt')
    max_seq_length = 20
    TokenizedSentence.setup(tokenizer, max_seq_length)
    #                                 0 123 45 6 78 9
    example = pd.Series({'sentence': '宝马X3是最畅销的车。', 'prefix': '宝马X3', 'labels': {(5, 10): 'D', (9, 10): 'T', (6, 8): 'T'}})

    # for multi-head
    labels = ['T', 'D', 'S', 'E']
    label_map = {l: i for i, l in enumerate(labels)}
    feature = convert_example_to_feature(0, example, label_map, task='multihead_labeling')
    print(feature)

    # for normal sequence labeling
    labels = ['O', 'B-T', 'I-T', 'B-D', 'I-D', 'B-S', 'I-S', 'B-E', 'I-E']
    label_map = {l: i for i, l in enumerate(labels)}
    feature = convert_example_to_feature(0, example, label_map, task='labeling')
    print(feature)
