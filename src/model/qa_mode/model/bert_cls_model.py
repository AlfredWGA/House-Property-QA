# coding=utf-8

import torch.nn as nn
from torch.nn import  MSELoss,CrossEntropyLoss
from transformers import BertModel, BertConfig

class BertCLSModel(nn.Module):
    def __init__(self, bert_model_dir, args):
        super(BertCLSModel, self).__init__()
        self.config = BertConfig.from_pretrained(bert_model_dir)
        self.config.output_hidden_states = False
        self.config.output_attentions = False
        self.bert = BertModel.from_pretrained(bert_model_dir, config=self.config)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, args.class_num)
        self.class_num = args.class_num


    def forward(self, batch_data):


        tokens_tensor, segments_tensors, att_mask, _,_,_, labels = batch_data
        outputs = self.bert(tokens_tensor, attention_mask=att_mask, token_type_ids=segments_tensors)
        class_encode = outputs[1]
        class_encode = self.dropout(class_encode)
        o = self.fc1(class_encode)
        out = (o, o)

        return out

    def get_loss_function(self):
        if self.class_num == 1:
            return MSELoss()
        else:
            return CrossEntropyLoss()
