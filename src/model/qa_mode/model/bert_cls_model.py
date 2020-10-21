# coding=utf-8

import torch.nn as nn
from torch.nn import  MSELoss, CrossEntropyLoss, BCELoss
from transformers import BertModel, BertConfig
import torch.nn.functional as F
import torch
import os

class BertCLSModel(nn.Module):
    def __init__(self, bert_model_dir, args):
        super(BertCLSModel, self).__init__()
        self.bert_model_dir = bert_model_dir
        self.config = BertConfig.from_pretrained(bert_model_dir)
        self.config.output_hidden_states = False
        self.config.output_attentions = False
        self.bert = BertModel.from_pretrained(bert_model_dir, config=self.config)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(4*self.bert.config.hidden_size, args.class_num)
        self.class_num = args.class_num
        self.sigmod = nn.Sigmoid()


    def reset_param(self):
        self.bert = BertModel.from_pretrained(self.bert_model_dir).to('cuda')
        self.fc1.reset_parameters()


    def forward(self, batch_data):


        tokens_tensor, segments_tensors, att_mask, _,_,_, labels = batch_data
        outputs = self.bert(tokens_tensor, attention_mask=att_mask, token_type_ids=segments_tensors)
        seq_hidden = outputs[0] # [b, seq, hid]

        q, _ = torch.max(seq_hidden, dim=1) # [b, hid]
        a = torch.mean(seq_hidden, dim=1) # [b, hid]
        t = seq_hidden[:, -1] # [b, hid]
        e = seq_hidden[:, 0] # [b, hid]

        class_encode = torch.cat([q, a, t, e], dim=-1) # [b, 4*hid]

        class_encode = self.dropout(class_encode)
        logit = self.fc1(class_encode)
        pre = self.sigmod(logit)
        out = (pre, pre)

        return out

    def get_loss_function(self):
        if self.class_num == 1:
            # return MSELoss()
            return BCELoss()
        else:
            return CrossEntropyLoss()
