import unittest
import pandas as pd
import numpy as np

from tokenizers import BertWordPieceTokenizer

from tokenize_utils import TokenizedSentence
from utils import convert_example_to_feature, SeqLabelDataset
from predict import convert_model_output_to_entities, multihead_convert_model_output_to_entities


class TestUtils(unittest.TestCase):

    def test_convert_example_to_feature(self):
        tokenizer = BertWordPieceTokenizer('assets/bert-base-chinese-vocab.txt')
        TokenizedSentence.setup(tokenizer, max_length=20)

        # for multi-head
        #                                 0 123 45 6 78 9
        example = pd.Series({'sentence': '宝马X3是最畅销的车。', 'prefix': '宝马X3', 'labels': {(5, 10): 'D', (9, 10): 'T', (6, 8): 'T'}})

        labels = ['T', 'D', 'S', 'E']
        label_map = {l: i for i, l in enumerate(labels)}
        feature = convert_example_to_feature(0, example, label_map, task='multihead_labeling')
        print(feature)

        logits = np.array([feature.label_ids])
        logits[logits==-100] = 0
        batch_tokend = [TokenizedSentence(example.sentence, prefix=example.prefix)]
        entities = multihead_convert_model_output_to_entities(batch_tokend, logits, labels, is_pred_ids=True)
        self.assertEqual(example.labels, entities[0])


        # for normal sequence labeling
        #                                 0 123 45 6 78 9
        example = pd.Series({'sentence': '宝马X3是最畅销的车。', 'prefix': '宝马X3', 'labels': {(5, 8): 'D', (9, 10): 'T'}})

        labels = ['O', 'B-T', 'I-T', 'B-D', 'I-D', 'B-S', 'I-S', 'B-E', 'I-E']
        label_map = {l: i for i, l in enumerate(labels)}
        feature = convert_example_to_feature(0, example, label_map, task='labeling')
        print(feature)

        logits = np.array([feature.label_ids])
        logits[logits==-100] = 0
        batch_tokend = [TokenizedSentence(example.sentence, prefix=example.prefix)]
        entities = convert_model_output_to_entities(batch_tokend, logits, labels, is_pred_ids=True)
        self.assertEqual(example.labels, entities[0])
        

    def test_SeqLabelDataset(self):
        tokenizer = BertWordPieceTokenizer('assets/bert-base-chinese-vocab.txt')
        
        # for normal sequence labeling
        labels = ['O', 'B-P', 'I-P']

        dataset = SeqLabelDataset(
            task='labeling',
            data_dir='assets/data/sequence_labeling/',
            mode='train',
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=50,
        )

        print(dataset[0])

        # for multi-head sequence labeling
        labels = ['O', 'T', 'D', 'S']

        dataset = SeqLabelDataset(
            task='multi_head_labeling',
            data_dir='assets/data/multihead_sequence_labeling/',
            mode='train',
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=50,
        )

        print(dataset[0])
        