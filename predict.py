from typing import List, Dict, Tuple, Optional
from tokenize_utils import TokenizedSentence
import torch
import numpy as np
import re

from seq_utils import get_entities, filter_invalid


def batch_predict(model, batch_tokend: TokenizedSentence, device) -> np.ndarray:
    '''
        outputs: shape=(N, L, S) for labeling task, shape=(N, num_heads, L, 3) for multi-head labeling task 
    '''
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = {
            'input_ids': torch.tensor([t.input_ids for t in batch_tokend], dtype=torch.long, device=device),
            'attention_mask': torch.tensor([t.attention_mask for t in batch_tokend], dtype=torch.long, device=device),
            'token_type_ids': torch.tensor([t.token_type_ids for t in batch_tokend], dtype=torch.long, device=device),
        }

        outputs = model(**inputs)
        outputs = outputs[0].detach().cpu().numpy()

    return outputs


def convert_model_output_to_entities(batch_tokend, logits, labels):
    '''
        logits: np.array with shape (B, L, S)
        输出：[
                {(start, end): tag, (start, end): tag}
             ]
    '''
    all_pred_ids = logits.argmax(axis=-1)
    all_entities = []
    for tokend, pred_ids in zip(batch_tokend, all_pred_ids):
        mask_len = sum(tokend.attention_mask)
        pred_ids = pred_ids[:(mask_len-1)]  # remove pad and tailing [SEP]
        pred_labels = [labels[x] for x in pred_ids]
        start_pos = tokend.token_type_ids.index(1) if tokend.prefix else 0
        # 从开头到 start_pos 这个区间，是无效区间
        entities = get_entities(pred_labels)
        entities = filter_invalid(entities, [(0, start_pos-1)])

        pred_entities = {}
        for tag, start, end in entities:
            span = tokend.token_span_to_char_span((start, end+1))
            pred_entities[span] = tag
        
        all_entities.append(pred_entities)
    
    return all_entities



