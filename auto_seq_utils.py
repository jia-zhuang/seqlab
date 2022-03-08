# refer to seqeval
import re


PAT = re.compile(r'B+')

def get_entities(seq, suffix=False):
    seq = ''.join([x[0] for x in seq])
    
    chunks = []
    pos = 0
    while True:
        match = PAT.search(seq, pos)
        if match is None: break
        
        start, end = match.span()
        chunks.append(('A', start, end - 1))
        
        pos = end
    
    return chunks

'''
def get_entities(seq, suffix=False):
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
        
    prev_tag = 'O'
    prev_type = ''
    begin_offset = None
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_) and begin_offset is not None:
            chunks.append((prev_type, begin_offset, i-1))
            begin_offset = None
        
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        
        prev_tag = tag
        prev_type = type_
    
    return chunks
'''


def start_of_chunk(prev_tag, tag):
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    return chunk_start


def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_type != type_:
        chunk_end = True

    return chunk_end


def span_overlap(a, b):
    ''' 判断两个闭区间是否有重叠部分
        a, b: (start, end) 闭区间
    '''
    if a <= b:
        return a[1] >= b[0]
    else:
        return b[1] >= a[0]


def filter_invalid(entities, invalid_spans=None):
    ''' 过滤掉与 invalid_spans 有重叠的实体
        entities: 格式为 [('M', 0, 0), ('A', 2, 4), ('X', 7, 7)]，闭区间
        invalid_spans: 指定要过滤的区间，格式为 [(0, 1), (7, 7)], 闭区间
    '''

    if invalid_spans is None:
        return entities
    
    res = []
    for tag, start, end in entities:
        for span in invalid_spans:
            if span_overlap(span, (start, end)):
                break
        else:
            res.append((tag, start, end))
    
    return res


if __name__ == "__main__":
    # 测试
    seq = ['B-A', 'I-A', 'I-A', 'O', 'O', 'B-X']
    print(seq)
    print(get_entities(seq))
    print()

    seq = ['O', 'I-A', 'I-A', 'O', 'O', 'B-X']
    print(seq)
    print(get_entities(seq))
    print()

    seq = ['B-A', 'I-C', 'I-A', 'O', 'O', 'B-X']
    print(seq)
    print(get_entities(seq))
    print()

    seq = ['B-A', 'O', 'I-A', 'O', 'O', 'B-X']
    print(seq)
    print(get_entities(seq))
    print()
    
    seq = ['B-A', 'B-A', 'I-A', 'O', 'O', 'B-X']
    print(seq)
    print(get_entities(seq))
    print()
