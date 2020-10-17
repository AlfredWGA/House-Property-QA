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
RAW_DATA = os.path.join(DATA_DIR, 'raw_data', 'law')

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

        for qid, did, query, q_len, text, text_len, _, label in batch_data:
            seq = query + text
            pad_num = self.args.max_seq_len - q_len - text_len - 3
            mask = [1]*(q_len + text_len + 3) + [0]*(pad_num)
            type = [0]*(q_len +2) + [1]*(self.args.max_seq_len - q_len -2)
            assert len(seq) == len(mask) == len(type) == self.args.max_seq_len

            text_input.append(seq)
            mask_input.append(mask)
            type_input.append(type)
            y.append(label)

        text_input = np.array(text_input) # [b, seq_len]
        mask_input = np.array(mask_input) # [b, seq_len]
        type_input = np.array(type_input) # [b, seq_len]
        y = np.array(y) # [b]
        if self.args.class_num == 1:
            y = np.expand_dims(y, axis=-1) # [b, 1]
            y[y > 0.5] = 1.0  # 将标签从 0 1转化为 -1 1 , 然后输入 sigmod 函数，得到 0 1 输出
            y[y < 0.5] = -1.0

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
        self.buffer = Queue(maxsize=self.batch_size)
        self.full_sem = threading.Semaphore(0)
        self.empty_sem = threading.Semaphore(1)
        self.data_buff_thread = Thread(target=self.get_data_worker)
        self.data_buff_thread.start()

        self.flag_data_out = False



    def get_data_worker(self):

        self.flag_data_out = False
        i = 0
        batches = 0
        while True:
            qid, did, label = self.batcher.qd_pairs_id[i] # 得到 qid did label

            query = self.batcher.get_query(qid) # 得到 q 内容
            docs = self.batcher.get_doc(did) # 得到 d 内容
            if batches == 0:
                self.empty_sem.acquire() # 请求一个批次

            q_len = len(query)  # 得到query长度
            query = [CLS_TOKEN] + query + [SEP_TOKEN]
            query = self.tokenizer.convert_tokens_to_ids(query)  # 转化为id

            ''' 处理并缓存 q d label '''
            for doc in docs:
                d_len = len(doc)  # 得到doc长度

                ''' 保留完整的query, 对 document 进行截断 '''
                if d_len + q_len + 2 > self.args.max_seq_len - 1:
                    exceed_num = (d_len + q_len + 2 ) - self.args.max_seq_len + 1
                    doc = doc[:-exceed_num]
                    d_len = len(doc)

                    assert len(doc) + q_len + 2 == self.args.max_seq_len - 1,  ' %d %d %d' % (len(doc), q_len , exceed_num)

                doc = doc + [SEP_TOKEN]

                pad_num = self.args.max_seq_len - q_len - d_len - 3
                doc = doc + [PAD_TOKEN] * pad_num

                assert len(doc) + len(query) == self.args.max_seq_len == d_len + q_len + 3 + pad_num, ' %d %d %d %d' % (len(doc), len(query), d_len, q_len )

                text = self.tokenizer.convert_tokens_to_ids(doc)  # 将text转化为token

                self.buffer.put((qid, did, query, q_len, text, d_len, 0, label))
                batches += 1
                if batches >= self.batch_size:  # 每完成一个batch 释放一个信号量
                    self.full_sem.release()  # 释放一个batch信号量
                    batches = 0

            i += 1
            if i >= len(self.batcher.qd_pairs_id):  # 如果没处理满一个batch但处理完所有的数据
                i = 0
                self.flag_data_out = True
                self.full_sem.release()  # 完成最后一个batch，释放一个信号量
                batches = 0


    def __getitem__(self, item: int):

        self.full_sem.acquire()  # 请求一个批次

        if self.flag_data_out == True:

            self.flag_data_out = False
            batch_data = []
            remain = self.buffer.qsize()
            for _ in range(remain):
                batch_data.append(self.buffer.get())

            for tran in self.transform:
                batch_data = tran(batch_data)

        else:
            assert self.buffer.qsize() == self.batch_size
            batch_data = []
            for _ in range(self.batch_size):
                batch_data.append(self.buffer.get())

            for tran in self.transform:
                batch_data = tran(batch_data)

        self.empty_sem.release()  # 释放一个空批次

        return batch_data

    def __len__(self) -> int:

        num_batch = (len(self.batcher.qd_pairs_id) // self.args.batch_size)
        remain = len(self.batcher.qd_pairs_id) % self.args.batch_size
        if remain > 0:
            num_batch += 1

        return num_batch

    def on_epoch_end_callback(self, callback=None):
        """Reorganize the data_bert while epoch is ended."""
        if callback:
            callback()

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
        self.buffer = Queue(maxsize=self.batch_size)
        self.full_sem = threading.Semaphore(0)
        self.empty_sem = threading.Semaphore(1)
        self.data_buff_thread = Thread(target=self.get_data_worker)
        self.data_buff_thread.start()

        self.flag_data_out = False
        self.flag_data_out_once = False




    def get_data_worker(self):

        self.flag_data_out = False
        i = 0
        batches = 0
        while True:
            qid, did, label = self.batcher.qd_pairs_id[i]
            query = self.batcher.get_query(qid)
            docs = self.batcher.get_doc(did)

            q_len = len(query) # 得到query长度
            query = [CLS_TOKEN] + query + [SEP_TOKEN]
            query = self.tokenizer.convert_tokens_to_ids(query) # 转化为id

            ''' 处理并缓存 q d label '''

            for doc in docs:

                if batches == 0:
                    self.empty_sem.acquire()

                d_len = len(doc) # 得到doc长度

                ''' 保留完整的query, 对 document 进行截断 '''
                if d_len + q_len + 2 > self.args.max_seq_len - 1:
                    exceed_num = (d_len + q_len + 2 ) - self.args.max_seq_len + 1
                    doc = doc[:-exceed_num]
                    d_len = len(doc)

                    assert len(doc) + q_len + 2 == self.args.max_seq_len - 1,  '%s %d %d %d' % (self.dataset_name ,len(doc), q_len, exceed_num )

                doc = doc + [SEP_TOKEN]

                pad_num = self.args.max_seq_len - q_len - d_len - 3

                doc = doc + [PAD_TOKEN]*pad_num

                assert len(doc) + len(query) == self.args.max_seq_len == d_len + q_len + 3 + pad_num, ' %d %d %d %d' % (len(doc), len(query), d_len, q_len )

                text = self.tokenizer.convert_tokens_to_ids(doc)  # 将text转化为token

                self.buffer.put((qid, did, query, q_len, text, d_len, 0, label))

                batches += 1
                if batches >= self.batch_size: # 每完成一个batch 释放一个信号量
                    self.full_sem.release() # 释放一个信号量
                    batches = 0

            i += 1
            if i >= len(self.batcher.qd_pairs_id): # 如果没处理满一个batch但处理完所有的数据
                i = 0
                self.flag_data_out = True
                self.full_sem.release() # 完成最后一个batch，释放一个信号量
                batches = 0


    def __getitem__(self, item: int):


        self.full_sem.acquire() # 请求一个批次

        if self.flag_data_out == True:
            self.flag_data_out = False

            batch_data = []
            remain = self.buffer.qsize()
            for _ in range(remain):
                batch_data.append(self.buffer.get())

            for tran in self.transform:
                batch_data = tran(batch_data)

        else:
            assert self.buffer.qsize() == self.batch_size
            batch_data = []
            for _ in range(self.batch_size):
                batch_data.append(self.buffer.get())

            for tran in self.transform:
                batch_data = tran(batch_data)

        self.empty_sem.release() # 释放一个空批次

        return batch_data



    def __len__(self) -> int:

        num_batch = (len(self.batcher.qd_pairs_id) // self.args.batch_size)
        remain = len(self.batcher.qd_pairs_id) % self.args.batch_size
        if remain > 0:
            num_batch += 1

        return num_batch


    def on_epoch_end_callback(self, callback=None):
        """Reorganize the data_bert while epoch is ended."""
        if callback:
            callback()

    def get_all(self):

        return self.batcher.get_all()