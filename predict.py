import os
from typing import List, Dict, Tuple, Optional
from tokenize_utils import TokenizedSentence
import torch
import numpy as np

from tokenizers import BertWordPieceTokenizer
from transformers import BertPreTrainedModel
from transformers import BertForTokenClassification

from seq_utils import get_entities, filter_invalid
from multi_head import BertForMultiHeadTokenClassification


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


def convert_model_output_to_entities(batch_tokend, logits, labels, is_pred_ids=False):
    '''
        logits: np.array with shape (B, L, S)
        输出：[
                {(start, end): tag, (start, end): tag}
             ]
    '''
    if is_pred_ids:
        all_pred_ids = logits
    else:
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


def multihead_convert_model_output_to_entities(batch_tokend, logits, heads, is_pred_ids=False):
    '''
        logits: np.array with shape (B, head, L, S)
        输出：[
            {(start, end): tag, (start, end): tag}
        ]
    '''
    entities = []  # (num_heads, B)
    for i, _ in enumerate(heads):
        entities.append(
            convert_model_output_to_entities(batch_tokend, logits[:, i], ['O', 'B-E', 'I-E'], is_pred_ids)
        )

    outputs = []
    for record in zip(*entities):  # (N, num_heads)
        tmp = {}
        for head, span_map in zip(heads, record):
            tmp.update({span: head for span in span_map})
        outputs.append(tmp)
    
    return outputs


class Predictor:
    def __init__(self, model_path, labels, max_seq_length, multi_head=False):
        self.multi_head = multi_head
        self.labels_or_heads = labels  # labels means heads if multi_head=True

        # load model
        model_cls = BertForMultiHeadTokenClassification if multi_head else BertForTokenClassification
        self.model = model_cls.from_pretrained(model_path, num_labels=len(labels))

        # load tokenizer
        tokenizer = BertWordPieceTokenizer(os.path.join(model_path, 'vocab.txt'))
        TokenizedSentence.setup_tokenizer(tokenizer, max_seq_length=max_seq_length)
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def predict(self, data):
        '''
            data = [
                {              # 0 1 23 4 56 7 89 10 11 12 13 
                    'sentence': '姚明是中国最优秀的篮球运动员',
                    'prefix': '姚明',  # 或者为 None
                }
            ]

            outputs = [
                {
                    (3, 14): 'D',
                    (9, 14): 'T'
                }
            ]
        '''
        batch_tokend = [TokenizedSentence(ex['sentence'], prefix=ex['prefix']) for ex in data]
        
        logits = batch_predict(self.model, batch_tokend, self.device)

        if self.multi_head:  # logits.shape = (N, num_heads, L, 3)
            outputs = multihead_convert_model_output_to_entities(batch_tokend, logits, self.labels_or_heads)
        else:   # logits.shape = (N, L, S)
            outputs = convert_model_output_to_entities(batch_tokend, logits, self.labels_or_heads)
        
        return outputs


if __name__ == "__main__":
    # test
    import pandas as pd
    dev_df = pd.read_pickle('datasets/v1_v2_short/dev.pkl')
    examples = dev_df.head().to_dict(orient='records')
    print(examples)
    
    from utils import get_labels
    labels = get_labels('labels.txt')
    predictor = Predictor('models/debug7/', labels, max_seq_length=200, multi_head=False)
    preds = predictor.predict(examples)
    print(preds)
