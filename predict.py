from tokenize_utils import TokenizedSentence
import torch
import numpy as np
import re


PATTERN = re.compile(r'12*')
REV_PATTERN = re.compile(r'34*')


def get_label_span_by_pattern(pred_ids_str, pattern, start_pos=0):
    '''
        pred_ids_str: convert pred_ids to str, which is convenient for regex search
        pattern: regex pattern
        start_pos: 有prefix的情况下，通过设置strat_pos来跳过prefix部分
    '''
    spans = []
    pos = start_pos
    while True:
        match = pattern.search(pred_ids_str, pos)
        if match:
            span = match.span()
            spans.append(span)
            pos = span[1]
        else:
            break
    
    return spans


def predict_batch(model, batch, TokenizedSentence, device):
    '''
        batch: [{'sentence': '', 'prefix': ''}]
    '''
    model.to(device)

    batch_tokend = [TokenizedSentence(ex['sentence'], prefix=ex['prefix']) for ex in batch]

    inputs = {
        'input_ids': torch.tensor([t.input_ids for t in batch_tokend], dtype=torch.long, device=device),
        'attention_mask': torch.tensor([t.attention_mask for t in batch_tokend], dtype=torch.long, device=device),
        'token_type_ids': torch.tensor([t.token_type_ids for t in batch_tokend], dtype=torch.long, device=device),
    }

    with torch.no_grad():
        outputs = model(**inputs)
        outputs = outputs[0].detach().cpu().numpy()
    
    all_pred_ids = np.argmax(outputs, axis=-1)

    all_labels = []
    for tokend, pred_ids in zip(batch_tokend, all_pred_ids):
        pred_ids_str = ''.join(map(str, pred_ids))
        start_pos = 0 if tokend.prefix is None else tokend.token_type_ids.index(1) 
        spans = get_label_span_by_pattern(pred_ids_str, PATTERN, start_pos=start_pos)
        
        labels = [tokend.get_phrase_by_token_span(s) for s in spans]
        all_labels.append(labels)
    
    return all_labels
