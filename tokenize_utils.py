
'''
实现原始文本与tokenize后的文本的对齐功能
原理：通过映射
huggingface/tokenizers 已经实现了tokenize后的token对应原始文本的idx，
这里只要实现反过来的映射，即原始文本中每个char对应token的idx即可
'''


class TokenizedSentence:
    '''使用前需要先调用setup'''
    tokenizer = None
    max_length = None
    
    @classmethod
    def setup(cls, tokenizer, max_length):
        cls.tokenizer = tokenizer
        cls.tokenizer.enable_truncation(max_length=max_length)
        cls.tokenizer.enable_padding(length=max_length)
    
    def __init__(self, sentence=None, prefix=None):
        
        if sentence is not None:  
            if prefix is None:
                tokend = self.tokenizer.encode(sentence)
            else:
                tokend = self.tokenizer.encode(prefix, pair=sentence)
            
            self.sentence = sentence
            self.prefix = prefix
            self.tokens = tokend.tokens
            self.input_ids = tokend.ids
            self.attention_mask = tokend.attention_mask
            self.token_type_ids = tokend.type_ids
            self.token2char = tokend.offsets  # List[(int, int)] 每个token对应的char span

            self.char2token = [None] * len(sentence)
            sentence_type_id = 0 if self.prefix is None else 1
            for i, ((start, end), mask, type_id) in enumerate(zip(self.token2char, self.attention_mask, self.token_type_ids)):
                if mask == 1 and type_id == sentence_type_id:
                    self.char2token[start:end] = [i] * (end - start)
    
    def char_span_to_token_span(self, char_span):
        token_indexes = self.char2token[char_span[0]:char_span[1]]
        token_indexes = list(filter(None, token_indexes))
        if token_indexes:
            return token_indexes[0], token_indexes[-1] + 1  # [start, end)
        else:  # empty
            return 0, 0
    
    def token_span_to_char_span(self, token_span):
        char_indexes = self.token2char[token_span[0]:token_span[1]]
        char_indexes = [span for span in char_indexes if span != (0, 0)]  # 删除CLS/SEP对应的span
        start, end = char_indexes[0][0], char_indexes[-1][1]
        return start, end
    
    def get_phrase_char_span(self, phrase):
        start = self.sentence.find(phrase)
        if start >= 0:  # phrase in sentence
            return start, start + len(phrase)
        else:  # phrase not in sentence
            return 0, 0
    
    def get_phrase_by_token_span(self, token_span):
        start, end = self.token_span_to_char_span(token_span)
        return self.sentence[start: end]
    
    def dump_to_dict(self):
        return {
            'sentence': self.sentence,
            'prefix': self.prefix,
            'tokens': self.tokens,
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'token_type_ids': self.token_type_ids,
            'token2char': self.token2char,
            'char2token': self.char2token
        }
    
    @classmethod
    def load_from_dict(cls, data):
        '''data: dict, 格式参考`dump_to_dict`'''
        instance = cls()
        for k, v in data.items():
            setattr(instance, k, v)
        return instance


if __name__ == "__main__":
    from tokenizers import BertWordPieceTokenizer
    # tests
    vocab_file = 'assets/bert-base-chinese-vocab.txt'
    tokenizer = BertWordPieceTokenizer(vocab_file)
    TokenizedSentence.setup(tokenizer, max_length=200)
    sent = "埃尔温·薛定谔（Erwin Schrödinger，1887年8月12日—1961年1月4日），男，奥地 利物理学家，量子力学奠基人之一"
    phrase = '奥地 利'
    char_span = (50, 54)

    tokend_sent = TokenizedSentence(sentence=sent)
    token_span = tokend_sent.char_span_to_token_span(char_span)
    assert tokend_sent.get_phrase_by_token_span(token_span) == phrase

    prefix = 'Erwin Schrödinger'
    tokend_sent = TokenizedSentence(sentence=sent, prefix=prefix)
    token_span = tokend_sent.char_span_to_token_span(char_span)
    assert tokend_sent.get_phrase_by_token_span(token_span) == phrase

    print('Tests passed!')