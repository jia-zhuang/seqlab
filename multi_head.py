import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class BertForMultiHeadTokenClassification(BertPreTrainedModel):
    ''' 用 Bert 做多头序列标注，比如：
        [CLS] 姚 明 [SEP] 姚 明 是 中 国 最 优 秀 的 篮 球 运 动 员 [SEP]
          X   X  X   X   O  O  O   B  I  I  I  I  I  I  I I  I   X      head0(描述)
          X   X  X   X   O  O  O   O  O  O  O  O  B  I  I  I  I  X      head1(标签)
          ...
          ...                                                           headn
    '''
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels    # num of heads

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.clfs = nn.ModuleList(
            [nn.Linear(config.hidden_size, 3) for _ in range(config.num_labels)]  # 3 for B, I, O
        )

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        '''
        input_ids, attention_mask, token_type_ids: B x L
        labels: B x num_heads x L
        '''
        
        outputs = self.bert(input_ids, attention_mask, token_type_ids)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = ()
        for clf in self.clfs:
            logits += (clf(sequence_output),)    # B x L x 3 for each head

        logits = torch.stack(logits, dim=1)  # B x num_heads x L x 3

        outputs = (logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 在数据准备的时候，padding 部分已经通过 ignore_index=-100 处理过了，这里可以直接算loss
            active_logits = logits.view(-1, 3)   # B*num_heads*L x 3
            active_labels = labels.view(-1)   # B*num_heads*L

            loss = loss_fct(active_logits, active_labels)

            outputs = (loss,) + outputs
        
        return outputs
