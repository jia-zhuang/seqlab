import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers import torch_distributed_zero_first

from tokenize_utils import TokenizedSentence

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
    label_ids: Optional[List[int]] = None


class SeqLabelDataset(Dataset):
    def __init__( self, data_dir, mode, tokenizer, labels, max_seq_length,
        overwrite_cache=False, local_rank=-1):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length)),
        )

        with torch_distributed_zero_first(local_rank):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                # read examples
                examples = pd.read_pickle(os.path.join(data_dir, f'{mode}.pkl'))
                # convert examples to features
                TokenizedSentence.setup_tokenizer(tokenizer, max_seq_length)
                label_map = {label: i for i, label in enumerate(labels)}
                
                self.features = []
                for ex_idx, example in enumerate(examples.itertuples()):
                    if ex_idx % 10000 == 0:
                        logger.info("Writing example %d of %d", ex_idx, examples.shape[0])

                    self.features.append(
                        convert_example_to_feature(ex_idx, example, TokenizedSentence, label_map)
                    )

                if local_rank in [-1, 0]:
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeature:
        return self.features[i]


'''
def generate_label_ids(token_spans, max_len, active_len, ignore_index=-100):
    label_ids = [0] * active_len + [ignore_index] * (max_len - active_len)
    for span in token_spans:
        if span[0] < span[1]:
            label_ids[span[0]] = 1  # B
            label_ids[(span[0] + 1): span[1]] = [2] * (span[1] - span[0] - 1)  # I
    return label_ids
'''

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


def convert_example_to_feature(ex_idx, example, TokenizedSentence, label_map):
    ''' 先不考虑span重叠的情况
    '''
    prefix = getattr(example, 'prefix', None)
    tokend_sent = TokenizedSentence(example.sentence, prefix=prefix)

    token_spans = {}
    for span, tag in example.labels.items():
        token_spans[tokend_sent.char_span_to_token_span(span)] = label_map[f'B-{tag}']  # 这里只用B-tag标识，I-tag通过+1获得，这就要求在写labels.txt文件时，遵循相邻和有序原则

    if prefix is None:
        ignore_mask = tokend_sent.attention_mask  # 无prefix(单句)的情况，attention_mask==0正好时需要被ignore的token
    else:
        ignore_mask = tokend_sent.token_type_ids  # 有prefix(上下句)的情况，token_type_ids==0正好时需要被ignore的token
    
    label_ids = generate_label_ids(token_spans, ignore_mask)

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
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
