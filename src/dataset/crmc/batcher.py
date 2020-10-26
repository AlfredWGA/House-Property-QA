# coding=utf-8


import logging
import os
import csv
import gzip
import json
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("Batcher")
logger.setLevel(logging.DEBUG)

import numpy as np
import matchzoo as mz
import pandas as pd
from tqdm import tqdm
import random
import pickle

script_abs_path = os.path.dirname(__file__)
ROOT_DIR = os.path.join(script_abs_path, '../../../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA = os.path.join(DATA_DIR, 'crmc2018')

train_path = os.path.join(RAW_DATA, 'crmc2018_train.csv')
test_path = os.path.join(RAW_DATA, 'crmc2018_dev.csv')

train_file = os.path.join(RAW_DATA, 'train.pickle')
test_file = os.path.join(RAW_DATA, 'test.pickle')

class Batcher():
    def __init__(self, dataset_name, tokenizer, args):

        self.dataset_name = dataset_name
        self.fold_num = 5

        self.args = args
        self.tokenizer = tokenizer
        self.total_sample_num = 0
        self.total_sample_label = None
        self.total_sample_list = []

        self.train_set_list = []
        self.test_set_list = []
        self.kf = KFold(n_splits=self.fold_num, random_state=args.seed)



        ''' 得到 qd_pairs_id '''
        self.instances = []
        self.max_seq_len = args.max_seq_len
        self.batches = []

        ''' 加载训练集 '''
        if not dataset_name == 'dev':
            if os.path.exists(train_file):
                with open(train_file, 'rb') as f:
                    self.total_sample_list = pickle.load(f)
                logging.info(f'Load processed training data from {train_file}.')
            else:
                self.df_train = pd.read_csv(train_path)

                for _, instance in tqdm(self.df_train.iterrows()):
                    qid, did, q, a, label = instance.query_id, instance.context_id ,instance.query, instance.answer, instance.label
                    ids, masks, segments = self._convert_to_transformer_inputs(q, a, self.max_seq_len)
                    assert len(ids) ==  self.max_seq_len
                    assert len(masks) ==  self.max_seq_len
                    assert len(segments) ==  self.max_seq_len
                    self.total_sample_list.append((ids, masks, segments, qid, did, label))
                with open(train_file, 'wb') as f:
                    pickle.dump(self.total_sample_list, f)

            self.total_sample_num = len(self.total_sample_list)
            _, _, _, _, _, self.total_sample_label = zip(*self.total_sample_list)

            self.get_k_fold_set()
            self.set_fold(0)

        else:   # 加载验证集
            if os.path.exists(test_file):
                with open(test_file, 'rb') as f:
                    self.instances = pickle.load(f)
                logging.info(f'Load testing training data from {test_file}.')
            else:
                self.df_test = pd.read_csv(test_path)
                for _, instance in tqdm(self.df_test.iterrows()):
                    qid, did, q, a = qid, did, q, a, label = instance.query_id, instance.context_id ,instance.query, instance.answer
                    ids, masks, segments = self._convert_to_transformer_inputs(q, a, self.max_seq_len)
                    assert len(ids) ==  self.max_seq_len
                    assert len(masks) ==  self.max_seq_len
                    assert len(segments) ==  self.max_seq_len
                    self.instances.append((ids, masks, segments, qid, did, 1))
                with open(test_file, 'wb') as f:
                    pickle.dump(self.instances, f)
            
            self.total_sample_num = len(self.instances)

            self.batches = self.get_batches()


    def get_k_fold_set(self):
        for train_index, test_index in self.kf.split(self.total_sample_list):
            self.train_set_list.append(train_index)
            self.test_set_list.append(test_index)

        tmp = []
        for indexes in self.test_set_list:
            tmp.extend(indexes)
        assert len(set(tmp)) == len(self.total_sample_list)


    ''' 设置使用第一fold的训练和数据集 '''
    def set_fold(self, fold_no):
        train_data = []
        test_data = []
        for i, inst in enumerate(self.total_sample_list):
            if i in self.train_set_list[fold_no]:
                train_data.append(inst)
            elif i in self.test_set_list[fold_no]:
                test_data.append(inst)
        if self.dataset_name == 'train':
            self.instances = train_data
        elif self.dataset_name == 'test':
            self.instances = test_data
        else:
            RuntimeError('Invalid dataset name !!')

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
            truncation=truncation_strategy,
            padding='max_length'
            )
            # print(inputs.keys())
            input_ids = inputs["input_ids"]
            input_segments = inputs["token_type_ids"]
            input_masks = inputs['attention_mask']

            return [input_ids, input_masks, input_segments]

        input_ids_q, input_masks_q, input_segments_q = return_id(question, answer, 'longest_first', max_sequence_length)

        return input_ids_q, input_masks_q, input_segments_q


if __name__ == '__main__':
    batcher = Batcher(dataset_name='test', tokenizer=None, args=None)
