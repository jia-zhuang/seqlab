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


class SeqLabDataset(Dataset):
    def __init__( self, data_dir, mode, tokenizer, labels, max_seq_length,
        overwrite_cache=False, local_rank=-1):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
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
                label_map = {label: i for i, label in enumerate(label_list)}
                
                self.features = []
                for example in examples.itertuples():
                    self.features.append(
                        convert_example_to_feature(example, TokenizedSentence, label_map)
                    )
                
                if local_rank in [-1, 0]:
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def generate_label_ids(token_spans, max_len, active_len, ignore_index=-100):
    label_ids = [0] * active_len + [ignore_index] * (max_len - active_len)
    for span in token_spans:
        if span[0] < span[1]:
            label_ids[span[0]] = 1  # B
            label_ids[(span[0] + 1): span[1]] = [2] * (span[1] - span[0] - 1)  # I
    return label_ids


def is_null(val):
    return val == '' or val == 'n/a'


def convert_example_to_feature(example, TokenizedSentence, label_map):
    tokend_sent = TokenizedSentence(example.sentence)
    subjects = [x['O'] for x in example.spos]
    subjects = [x for x in subjects if not is_null(x)]
    subjects = [x for x in subjects if x in example.sentence]

    sub_spans = []
    for sub in subjects:
        sub_spans.append(
            tokend_sent.char_span_to_token_span(
                tokend_sent.get_phrase_char_span(sub)
            )
        )
    
    label_ids = generate_label_ids(sub_spans, len(tokend_sent.input_ids), sum(tokend_sent.attention_mask))

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