# Bert 序列标注

## 简介

使用 Bert 做序列标注，使用 BIO 标注方式，支持常规序列标注和 Multi-Head 序列标注。

### 常规序列标注

```
tokens: [CLS] prefix [SEP] sentence [SEP]
labels:   X     X      X      BIO     X
```

- 标注始终在 sentence 上做，`X` 表示计算 `loss` 时会忽略这部分(通过 pytorch 的 `nn.CrossEntropyLoss().ignore_index` 实现)
- `prefix` 用来输入额外信息(可理解为有 condition 的序列标注)，比如：
    ```
    [CLS] 姚明 [SEP] 姚明是篮球运动员，普金是总统 [SEP]
    ```
    通过设置条件 `prefix=姚明`，可以只把 `篮球运动员` 标注出来，而不标注 `总统`。
    `prefx=None` 时，就是常规的序列标注。
- `prefix` 用法很灵活，再比如：
    ```
    [CLS] 小米 雷军 [SEP] 小米的董事长是雷军 [SEP]
    ```
    通过设置条件 `prefix` 为实体对，可以用来在句子中抽取实体对的关系。

### Multi-Head 序列标注

```
tokens: [CLS] prefix [SEP] sentence [SEP]
head 1:   X     X      X      BIO     X    # 比如用来表示 标签
head 2:   X     X      X      BIO     X    # 比如用来表示 描述
head 3:   X     X      X      BIO     X    # 比如用来表示 近义项
...
```

- 在多头上同时标注，可以处理标注的内容重叠的情况，比如：
    ```
    [CLS] 姚明 [SEP] 姚明是中国最优秀的篮球运动员。[SEP]
      X    X    X                  B  I   I    X     # head 1，表示“标签”  
      X    X    X         B   I   I   I   I    X     # head 2，表示“描述”  
    ```
    可以同时标注出 `(姚明, 标签, 篮球运动员)` 和 `(姚明, 标签, 中国最优秀的篮球运动员)`。

## Schema

- 输入

```json
[   # batch 形式输入
    {
        "sentence": "姚明是中国最优秀的篮球运动员",  
        "prefix": "姚明",     # 不需要时可以设置成 None
        "labels": {(9, 15): "T", (3, 15): "D"}   # 监督标签，训练时需要，预测时不需要
    }
]
```

- 输出

```json
[
    {{(9, 15): "T", (3, 15): "D"}}
]
```

## 使用

### 训练

- 参考 `run.sh`
- 需准备并检查以下文件
    
    - huggingface/transfomers Bert 预训练语言模型；
    - 训练/评测数据集：`{train,dev}.pkl`
    - 标签或头信息：`labels.txt`

### 预测

- 做来封装，可直接使用，参考 `predict.py`

