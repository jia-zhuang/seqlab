import unittest

from tokenizers import BertWordPieceTokenizer

from tokenize_utils import TokenizedSentence


class TestTokenizeUtils(unittest.TestCase):

    def test_TokenizedSentence(self):
        tokenizer = BertWordPieceTokenizer('assets/bert-base-chinese-vocab.txt')
        TokenizedSentence.setup(tokenizer, max_length=200)

        sent = "埃尔温·薛定谔（Erwin Schrödinger，1887年8月12日—1961年1月4日），男，奥地 利物理学家，量子力学奠基人之一"
        phrase = '奥地 利'
        char_span = (50, 54)

        tokend_sent = TokenizedSentence(sentence=sent)
        token_span = tokend_sent.char_span_to_token_span(char_span)
        self.assertEqual(phrase, tokend_sent.get_phrase_by_token_span(token_span))

        prefix = 'Erwin Schrödinger'
        tokend_sent = TokenizedSentence(sentence=sent, prefix=prefix)  # 不管有无 prefix，char_span 都是相对于 sentence 的
        token_span = tokend_sent.char_span_to_token_span(char_span)
        self.assertEqual(phrase, tokend_sent.get_phrase_by_token_span(token_span))
