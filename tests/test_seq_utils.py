import json
import logging
import unittest
from seq_utils import get_entities, filter_invalid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSeqUtils(unittest.TestCase):
    
    def test_get_entities(self):
        seq = ['B-A', 'I-A', 'I-A', 'O', 'O', 'B-X']
        ent = get_entities(seq)
        print('Test `get_entities` func ...')
        print('seq: ', seq)
        print('ent: ', ent)
        print()
        self.assertEqual([('A', 0, 2), ('X', 5, 5)], ent)

        seq = ['O', 'I-A', 'I-A', 'O', 'O', 'B-X']
        ent = get_entities(seq)
        print('Test `get_entities` func ...')
        print('seq: ', seq)
        print('ent: ', ent)
        print()
        self.assertEqual([('X', 5, 5)], ent)

        seq = ['B-A', 'I-C', 'I-A', 'O', 'O', 'B-X']
        ent = get_entities(seq)
        print('Test `get_entities` func ...')
        print('seq: ', seq)
        print('ent: ', ent)
        print()
        self.assertEqual([('A', 0, 0), ('X', 5, 5)], ent)

        seq = ['B-A', 'O', 'I-A', 'O', 'O', 'B-X']
        ent = get_entities(seq)
        print('Test `get_entities` func ...')
        print('seq: ', seq)
        print('ent: ', ent)
        print()
        self.assertEqual([('A', 0, 0), ('X', 5, 5)], ent)

    def test_filter_invalid(self):
        ent = [('M', 0, 0), ('A', 2, 4), ('X', 7, 7)]  # 闭区间
        span = [(0, 1), (7, 7)]  # 闭区间
        filtered_ent = filter_invalid(ent, span)
        print('Test `filter_invalid` func ...')
        print('ent: ', ent)
        print('span: ', span)
        print('filtered_ent: ', filtered_ent)
        print()
        self.assertEqual([('A', 2, 4)], filtered_ent)
