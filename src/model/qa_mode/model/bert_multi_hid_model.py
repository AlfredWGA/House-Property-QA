# coding=utf-8

import torch.nn as nn
from torch.nn import  MSELoss, CrossEntropyLoss, BCELoss
from transformers import BertModel, BertConfig
import torch.nn.functional as F
import torch
import os


class BertMultiHidModel(nn.Module):
    def __init__(self, bert_model_dir, args):
        super(BertMultiHidModel, self).__init__()
        self.bert_model_dir = bert_model_dir
        self.config = BertConfig.from_pretrained(bert_model_dir)
        self.config.output_hidden_states = True
        self.config.output_attentions = False
        self.hidden = self.config.hidden_size
        self.bert = BertModel.from_pretrained(bert_model_dir, config=self.config)
        self.att_w = nn.Linear(4*self.hidden, self.hidden)

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(4*self.hidden, args.class_num)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmod = nn.Sigmoid()
        self.class_num = args.class_num



    def forward(self, batch_data):

        tokens_tensor, segments_tensors, att_mask, _,_,_, labels = batch_data
        outputs = self.bert(tokens_tensor, attention_mask=att_mask, token_type_ids=segments_tensors)
        hiddens = outputs[2] # 12*[b, seq, hid]
        cls = outputs[1] # [b, hid]
        # cls = self.dropout(cls)  # [b, hid]

        last_first_hidden = hiddens[-1]
        last_second_hidden = hiddens[-2]
        # last_third_hideen = hiddens[-3]

        q1, _ = torch.max(last_first_hidden, dim=1) # [b, hid]
        a1 = torch.mean(last_first_hidden, dim=1) # [b, hid]
        t1 = last_first_hidden[:, -1] # [b, hid]
        e1 = last_first_hidden[:, 0] # [b, hid]

        q2, _ = torch.max(last_second_hidden, dim=1) # [b, hid]
        a2 = torch.mean(last_second_hidden, dim=1) # [b, hid]
        t2 = last_second_hidden[:, -1] # [b, hid]
        e2 = last_second_hidden[:, 0] # [b, hid]

        # q3, _ = torch.max(last_third_hideen, dim=1) # [b, hid]
        # a3 = torch.mean(last_third_hideen, dim=1) # [b, hid]
        # t3 = last_third_hideen[:, -1] # [b, hid]
        # e3 = last_third_hideen[:, 0] # [b, hid]

        class_encode1 = torch.cat([q1, a1, t1, e1], dim=-1) # [b, 4*hid]
        class_encode2 = torch.cat([q2, a2, t2, e2], dim=-1) # [b, 4*hid]
        # class_encode3 = torch.cat([q3, a3, t3, e3], dim=-1)  # [b, 4*hid]
        class_encode1 = self.dropout(class_encode1)
        class_encode2 = self.dropout(class_encode2)
        # class_encode3 = self.dropout(class_encode3)


        att1 = self.tanh(self.att_w(class_encode1)) # [b, hid]
        att2 = self.tanh(self.att_w(class_encode2)) # [b, hid]
        # att3 = self.tanh(self.att_w(class_encode3))  # [b, hid]
        a1 = (cls*att1).sum(-1, keepdim=True) # [b, 1]
        a2 = (cls*att2).sum(-1, keepdim=True) # [b, 1]
        # a3 = (cls*att3).sum(-1, keepdim=True)  # [b, 1]

        # a2 = self.softmax(torch.cat([a2, a3], dim=-1)).unsqueeze(-1) # [b, 2, 1]
        # class_encode2 = torch.stack([class_encode2, class_encode3], dim=1) # [b, 2, 4*hid]
        # class_encode2 = (class_encode2*a2).sum(1) #  [b, 4*hid]
        # att2 = self.tanh(self.att_w(class_encode2)) # [b, hid]
        # a2 = (cls*att2).sum(-1, keepdim=True) # [b, 1]


        alpha = self.softmax(torch.cat([a1, a2], dim=-1)) # [b, 2]
        alpha = alpha.unsqueeze(-1) # [b, 2, 1]
        class_encode = torch.stack([class_encode1, class_encode2], dim=1) # [b, 2, 4*hid]
        class_encode = (class_encode*alpha).sum(1) # [b, 4*hid]

        class_encode = self.dropout(class_encode)
        logit = self.fc1(class_encode)

        pre = self.sigmod(logit)
        out = (logit, pre)

        return out

    def get_loss_function(self):
        if self.class_num == 1:
            return MSELoss()
            # return BCELoss()
        else:
            return CrossEntropyLoss()


class BertMultiHidModel2(nn.Module):
    def __init__(self, bert_model_dir, args):
        super(BertMultiHidModel2, self).__init__()
        self.bert_model_dir = bert_model_dir
        self.config = BertConfig.from_pretrained(bert_model_dir)
        self.config.output_hidden_states = True
        self.config.output_attentions = False
        self.hidden = self.config.hidden_size
        self.bert = BertModel.from_pretrained(bert_model_dir, config=self.config)
        self.att_w = nn.Linear(4*self.hidden, self.hidden)

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(4*self.hidden, args.class_num)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmod = nn.Sigmoid()
        self.class_num = args.class_num



    def forward(self, batch_data):

        tokens_tensor, segments_tensors, att_mask, _,_,_, labels = batch_data
        outputs = self.bert(tokens_tensor, attention_mask=att_mask, token_type_ids=segments_tensors)
        hiddens = outputs[2] # 12*[b, seq, hid]
        cls = outputs[1] # [b, hid]

        last_first_hidden = hiddens[-1]
        last_second_hidden = hiddens[-2]


        q1, _ = torch.max(last_first_hidden, dim=1) # [b, hid]
        a1 = torch.mean(last_first_hidden, dim=1) # [b, hid]
        t1 = last_first_hidden[:, -1] # [b, hid]
        e1 = last_first_hidden[:, 0] # [b, hid]

        q2, _ = torch.max(last_second_hidden, dim=1) # [b, hid]
        a2 = torch.mean(last_second_hidden, dim=1) # [b, hid]
        t2 = last_second_hidden[:, -1] # [b, hid]
        e2 = last_second_hidden[:, 0] # [b, hid]


        class_encode1 = torch.cat([q1, a1, t1, e1], dim=-1) # [b, 4*hid]
        class_encode2 = torch.cat([q2, a2, t2, e2], dim=-1) # [b, 4*hid]

        class_encode1 = self.dropout(class_encode1)
        class_encode2 = self.dropout(class_encode2)


        att1 = self.tanh(self.att_w(class_encode1)) # [b, hid]
        att2 = self.tanh(self.att_w(class_encode2)) # [b, hid]

        a1 = (cls*att1).sum(-1, keepdim=True) # [b, 1]
        a2 = (cls*att2).sum(-1, keepdim=True) # [b, 1]

        alpha = self.softmax(torch.cat([a1, a2], dim=-1)) # [b, 2]
        alpha = alpha.unsqueeze(-1) # [b, 2, 1]
        class_encode = torch.stack([class_encode1, class_encode2], dim=1) # [b, 2, 4*hid]
        class_encode = (class_encode*alpha).sum(1) # [b, 4*hid]

        class_encode = self.dropout(class_encode)
        logit = self.fc1(class_encode)

        pre = self.sigmod(logit)
        out = (logit, pre)

        return out

    def get_loss_function(self):
        if self.class_num == 1:
            return MSELoss()
            # return BCELoss()
        else:
            return CrossEntropyLoss()









