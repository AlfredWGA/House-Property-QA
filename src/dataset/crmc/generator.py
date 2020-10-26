# coding=utf-8

import os
from .batcher import Batcher
from queue import Queue
from threading import Thread
import threading
import numpy as np
import torch
from tqdm import tqdm
import random

script_abs_path = os.path.dirname(__file__)
ROOT_DIR = os.path.join(script_abs_path, '../../../')
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
# RAW_DATA = os.path.join(DATA_DIR, 'raw_data', 'law')

import logging

logger = logging.getLogger("Generator")
logger.setLevel(logging.DEBUG)

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'


class ToBertInput():

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer


    def __call__(self, batch_data):
        text_input = []
        mask_input = []
        type_input = []
        y = []

        for ids, masks, segments, qid, did, label in batch_data:

            assert len(ids) == len(masks) == len(segments) == self.args.max_seq_len

            text_input.append(ids)
            mask_input.append(masks)
            type_input.append(segments)
            y.append(label)

        text_input = np.array(text_input) # [b, seq_len]
        mask_input = np.array(mask_input) # [b, seq_len]
        type_input = np.array(type_input) # [b, seq_len]
        y = np.array(y) # [b]
        if self.args.class_num == 1:
            y = np.expand_dims(y, axis=-1) # [b, 1]

        ''' 数据如果不足以并行，则需要重复 '''
        batch_size = y.shape[0]
        ratio = 1
        if batch_size > self.args.min_bz_per_gpu*self.args.gpu_num:
            over_num = batch_size % (self.args.min_bz_per_gpu*self.args.gpu_num)
        elif batch_size < self.args.min_bz_per_gpu*self.args.gpu_num:
            over_num = batch_size
        else:
            over_num = 0
        if over_num > 0:
            assert over_num % self.args.min_bz_per_gpu == 0
            remain_num = self.args.min_bz_per_gpu*self.args.gpu_num - over_num
            assert remain_num % self.args.min_bz_per_gpu == 0

            text_input = np.pad(text_input, ((0, remain_num),(0,0)), 'constant', constant_values=(1,1))  # [(b+remain_num), seq_len]
            mask_input = np.pad(mask_input, ((0, remain_num),(0,0)), 'constant', constant_values=(1,1))  # [(b+remain_num), seq_len]
            type_input = np.pad(type_input, ((0, remain_num),(0,0)), 'constant', constant_values=(1,1))  # [(b+remain_num), seq_len]
            if self.args.class_num == 1:
                y = np.pad(y, ((0, remain_num),(0,0)), 'constant', constant_values=(1, 1))  # [(b+remain_num), 1]
            else:
                y = np.pad(y, (0, remain_num), 'constant', constant_values=(1, 1))  # [(b+remain_num), 1]
            padded_batch_size = y.shape[0]
            ratio = batch_size/padded_batch_size

        text_input =  torch.Tensor(text_input).long()
        mask_inpit = torch.Tensor(mask_input).float()
        type_input = torch.Tensor(type_input).long()

        if self.args.class_num == 1:
            y = torch.Tensor(y).float()
        else:
            y = torch.Tensor(y).long()  # [b,]

        return (text_input, type_input, mask_inpit, y, y, y, y), ratio


class TrainDataGenerator():

    def __init__(self, dataset_model, dataset_name, tokenizer, args, transform=None):
        """Init."""
        self.transform = transform
        self.dataset_model = dataset_model
        self.dataset_name = dataset_name
        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        self.args = args
        self.max_q_len = self.args.max_q_len
        self.max_d_len = self.args.max_d_len
        self.max_seq_len = self.args.max_seq_len
        self.max_para_num = self.args.max_para_num
        self.batcher = Batcher(dataset_name=dataset_name,
                               args=args,
                               tokenizer=tokenizer)

    def set_fold(self, fold_no):
        self.batcher.set_fold(fold_no)

    def __getitem__(self, item: int):

        batch = self.batcher.batches[item]
        for trans in self.transform:
            batch = trans(batch)
        return batch

    def __len__(self) -> int:


        return len(self.batcher.batches)

    def get_all(self):
        return self.batcher.get_all()


class DevTestGenerator():

    def __init__(self, dataset_model, dataset_name, tokenizer, args, transform=None):
        """Init."""
        self.transform = transform
        self.dataset_model = dataset_model
        self.dataset_name = dataset_name
        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        self.args = args
        self.max_q_len = self.args.max_q_len
        self.max_d_len = self.args.max_d_len
        self.max_seq_len = self.args.max_seq_len
        self.max_para_num = self.args.max_para_num
        self.batcher = Batcher(dataset_name=dataset_name,
                               args=args,
                               tokenizer=tokenizer)

    def set_fold(self, fold_no):
        self.batcher.set_fold(fold_no)

    def __getitem__(self, item: int):

        batch = self.batcher.batches[item]
        for trans in self.transform:
            batch = trans(batch)
        return batch



    def __len__(self) -> int:

        return len(self.batcher.batches)



    def get_all(self):

        return self.batcher.get_all()