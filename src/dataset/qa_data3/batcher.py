# coding=utf-8


import logging
import os
import csv
import gzip
import json

from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("Batcher")
logger.setLevel(logging.DEBUG)

import numpy as np
import matchzoo as mz
import pandas as pd
from tqdm import tqdm
import random

script_abs_path = os.path.dirname(__file__)
ROOT_DIR = os.path.join(script_abs_path, '../../../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA = os.path.join(DATA_DIR, 'raw_data')

train_query_file = os.path.join(RAW_DATA, 'train', 'train.query.tsv')
train_reply_file = os.path.join(RAW_DATA, 'train', 'train.reply.tsv')

test_query_file = os.path.join(RAW_DATA, 'test', 'test.query.tsv')
test_reply_file = os.path.join(RAW_DATA, 'test', 'test.reply.tsv')


class Batcher():
    def __init__(self, dataset_name, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer

        ''' 得到 qd_pairs_id '''
        self.instances = []
        self.max_seq_len = args.max_seq_len
        self.batches = []

        if not dataset_name == 'dev':
            train_left = pd.read_csv(train_query_file, sep='\t', header=None)
            train_left.columns = ['id', 'q1']
            train_right = pd.read_csv(train_reply_file, sep='\t', header=None)
            train_right.columns = ['id', 'id_sub', 'q2', 'label']
            self.df_train = train_left.merge(train_right, how='left')  # 得到训练数据
            self.df_train['q2'] = self.df_train['q2'].fillna('NaN')

            instances = []
            for _, instance in tqdm(self.df_train[['q1', 'q2', 'label', 'id', 'id_sub']].iterrows()):
                qid, did, q, a, label = instance.id, instance.id_sub ,instance.q1, instance.q2, instance.label
                ids, masks, segments = self._convert_to_transformer_inputs(q, a, self.max_seq_len)
                assert len(ids) ==  self.max_seq_len
                assert len(masks) ==  self.max_seq_len
                assert len(segments) ==  self.max_seq_len
                instances.append((ids, masks, segments, qid, did, label))

            train_set, test_set = train_test_split(instances, test_size=0.2, random_state=42)

            if dataset_name == 'train':
                self.instances = train_set
            else:
                self.instances = test_set
        else:
            test_left = pd.read_csv(test_query_file, sep='\t', header=None, encoding='gbk')
            test_left.columns = ['id', 'q1']
            test_right = pd.read_csv(test_reply_file, sep='\t', header=None, encoding='gbk')
            test_right.columns = ['id', 'id_sub', 'q2']
            self.df_test = test_left.merge(test_right, how='left')  # 得到测试数据
            for _, instance in tqdm(self.df_test[['q1', 'q2', 'id', 'id_sub']].iterrows()):
                qid, did, q, a = instance.id, instance.id_sub, instance.q1, instance.q2
                ids, masks, segments = self._convert_to_transformer_inputs(q, a, self.max_seq_len)
                assert len(ids) ==  self.max_seq_len
                assert len(masks) ==  self.max_seq_len
                assert len(segments) ==  self.max_seq_len
                self.instances.append((ids, masks, segments, qid, did, 1))

        self.batches = self.get_batches()

    ''' 得到该批数据所有的 label '''
    def get_all(self):

        _, _, _, qid, did, Y = zip(*self.instances)
        X = {'id_left': np.array(qid),
             'id_right': np.array(did)}
        Y = np.array(Y)

        return X, Y

    def get_batches(self):
        batch_size = self.args.batch_size
        num_batch = (len(self.instances) // self.args.batch_size)
        remain = len(self.instances) % self.args.batch_size
        batches = []
        if remain > 0:
            num_batch += 1
        for i in range(num_batch):
            s_index = i*batch_size
            e_index = (i+1)*batch_size
            b = self.instances[s_index:e_index]
            batches.append(b)
        return batches


    def _convert_to_transformer_inputs(self, question, answer, max_sequence_length):
        """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
        def return_id(str1, str2, truncation_strategy, length):
            inputs = self.tokenizer.encode_plus(str1, str2,
                                           add_special_tokens=True,
                                           max_length=length,
                                           truncation=truncation_strategy)
            input_ids = inputs["input_ids"]
            input_masks = [1] * len(input_ids)
            input_segments = inputs["token_type_ids"]
            padding_length = length - len(input_ids)
            padding_id = self.tokenizer.pad_token_id
            input_ids = input_ids + ([padding_id] * padding_length)  # [max_seq]
            input_masks = input_masks + ([0] * padding_length)  # [max_seq]
            input_segments = input_segments + ([0] * padding_length)  # [max_seq]

            return [input_ids, input_masks, input_segments]

        input_ids_q, input_masks_q, input_segments_q = return_id(question, answer, 'longest_first', max_sequence_length)

        return input_ids_q, input_masks_q, input_segments_q


if __name__ == '__main__':
    batcher = Batcher(dataset_name='test', tokenizer=None, args=None)
