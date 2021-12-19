import os
from typing import List, Optional, Union

import numpy as np

from torch.utils.data.dataset import Dataset, IterableDataset


class SeqLabelDataset(IterableDataset):
    def __init__(self, task, data_dir, mode, is_small, tokenizer, labels, max_seq_length, overwrite_cache=False):
        self.data_file = os.path.join(data_dir, f'{mode}_small.txt' if is_small else f'{mode}.txt')
        logger.info(f"Creating features from dataset file at {self.data_file}")

        self.label_map = {label: i for i, label in enumerate(labels)}
        
        self.tokenizer = tokenizer
        self.tokenizer.enable_truncation(max_length=max_seq_length)
        self.tokenizer.enable_padding(length=max_seq_length)

    def __iter__(self):
        with open(self.data_file) as f:
            for line in f:
                text, dis_lst = line.strip().split('\x01')

                dis_lst = dis_lst.split(',')
                dis_lst = [self.label_map.get(x, 0) for x in dis_lst]
                labels = np.zeros(len(self.label_map), dtype=np.float32)
                labels[dis_lst] = 1.0

                tokend = self.tokenizer.encode(text)
                yield {'input_ids': tokend.ids, 'attention_mask': tokend.attention_mask, 'labels': labels}


def get_labels(path: str) -> List[str]:
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels

