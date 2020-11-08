# coding=utf-8


import logging
import os
import csv
import gzip
import json
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
RAW_DATA = os.path.join(DATA_DIR, 'raw_data')

train_query_file = os.path.join(RAW_DATA, 'train', 'train.query.tsv')
train_reply_file = os.path.join(RAW_DATA, 'train', 'train.reply.tsv')

test_query_file = os.path.join(RAW_DATA, 'test', 'test.query.tsv')
test_reply_file = os.path.join(RAW_DATA, 'test', 'test.reply.tsv')

train_query_bt_file = os.path.join(RAW_DATA, 'train', 'en','train.query.bt.tsv')
train_reply_bt_file = os.path.join(RAW_DATA, 'train', 'en', 'train.reply.bt.tsv')
en_preprocessed_data_dir = os.path.join(DATA_DIR, 'preprocessed_data',  'en')

train_query_bt_jp_file = os.path.join(RAW_DATA, 'train', 'jp','train.query.bt.jp.tsv')
train_reply_bt_jp_file = os.path.join(RAW_DATA, 'train', 'jp', 'train.reply.bt.jp.tsv')
jp_preprocessed_data_dir = os.path.join(DATA_DIR, 'preprocessed_data', 'jp')

train_query_neg_file = os.path.join(RAW_DATA, 'train', 'aug','train.query.aug4.tsv')
train_reply_neg_file = os.path.join(RAW_DATA, 'train', 'aug', 'train.reply.aug4.tsv')
neg_preprocessed_data_dir = os.path.join(DATA_DIR, 'preprocessed_data', 'aug')



class Batcher():
    def __init__(self, dataset_name, tokenizer, args):

        self.dataset_name = dataset_name
        self.fold_num = args.fold_num

        self.args = args
        self.tokenizer = tokenizer
        self.total_sample_num = 0
        self.total_sample_label = []
        self.total_sample_list = []

        self.total_sample_bt_list = []
        self.total_sample_bt_label = []

        self.train_set_list = []
        self.test_set_list = []

        self.kf = KFold(n_splits=self.fold_num, shuffle=True, random_state=args.seed)

        ''' 得到 qd_pairs_id '''
        self.instances = []
        self.max_seq_len = args.max_seq_len
        self.batches = []


        if not dataset_name == 'dev':
            if not os.path.exists(os.path.join(jp_preprocessed_data_dir, 'train_sample_list.pkl')):

                train_left = pd.read_csv(train_query_bt_jp_file, sep='\t', header=None)
                train_left.columns = ['id', 'q1', 'bt1']
                train_right = pd.read_csv(train_reply_bt_jp_file, sep='\t', header=None)
                train_right.columns = ['id', 'id_sub', 'q2', 'label', 'bt2']
                self.df_train = train_left.merge(train_right, how='left')  # 得到训练数据
                self.df_train['q2'] = self.df_train['q2'].fillna('你好')


                for _, instance in tqdm(self.df_train[['q1', 'q2', 'label', 'id', 'id_sub', 'bt1', 'bt2']].iterrows()):
                    qid, did, q, a, label = instance.id, instance.id_sub ,instance.q1, instance.q2, instance.label
                    q_bt = instance.bt1
                    a_bt = instance.bt2
                    ids, masks, segments = self._convert_to_transformer_inputs(q, a, self.max_seq_len)

                    assert len(ids) ==  self.max_seq_len
                    assert len(masks) ==  self.max_seq_len
                    assert len(segments) ==  self.max_seq_len
                    self.total_sample_list.append((ids, masks, segments, qid, did, label))

                    ids_bt, masks_bt, segments_bt = self._convert_to_transformer_inputs(q_bt, a_bt, self.max_seq_len)

                    assert len(ids_bt) ==  self.max_seq_len
                    assert len(masks_bt) ==  self.max_seq_len
                    assert len(segments_bt) ==  self.max_seq_len
                    self.total_sample_bt_list.append((ids_bt, masks_bt, segments_bt, qid, did, label))



                with open(os.path.join(jp_preprocessed_data_dir, 'train_sample_list.pkl'), 'wb') as f:
                    pickle.dump(self.total_sample_list ,f)
                with open(os.path.join(jp_preprocessed_data_dir, 'train_sample_bt_list.pkl'), 'wb') as f:
                    pickle.dump(self.total_sample_bt_list ,f)
                with open(os.path.join(jp_preprocessed_data_dir, 'df_train.pkl'), 'wb') as f:
                    pickle.dump(self.df_train, f)

            else:
                with open(os.path.join(jp_preprocessed_data_dir, 'train_sample_list.pkl'), 'rb') as f:
                    self.total_sample_list = pickle.load(f)
                with open(os.path.join(jp_preprocessed_data_dir, 'train_sample_bt_list.pkl'), 'rb') as f:
                    self.total_sample_bt_list  = pickle.load(f)
                with open(os.path.join(jp_preprocessed_data_dir, 'df_train.pkl'), 'rb') as f:
                    self.df_train = pickle.load(f)


            self.total_sample_num = len(self.total_sample_list)
            assert self.total_sample_num == len(self.total_sample_bt_list)
            _, _, _, _, _, self.total_sample_label = zip(*self.total_sample_list)
            _, _, _, _, _, self.total_sample_bt_label = zip(*self.total_sample_bt_list)


            self.get_k_fold_set()
            self.set_fold(0)

        else:
            if not os.path.exists(os.path.join(jp_preprocessed_data_dir, 'test_sample_list.pkl')):

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

                with open(os.path.join(jp_preprocessed_data_dir, 'test_sample_list.pkl'), 'wb') as f:
                    pickle.dump(self.instances ,f)
                with open(os.path.join(jp_preprocessed_data_dir, 'df_test.pkl'), 'wb') as f:
                    pickle.dump(self.df_test, f)
            else:
                with open(os.path.join(jp_preprocessed_data_dir, 'test_sample_list.pkl'), 'rb') as f:
                    self.instances = pickle.load(f)
                with open(os.path.join(jp_preprocessed_data_dir, 'df_test.pkl'), 'rb') as f:
                    self.df_test = pickle.load(f)


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

        train_data_bt = []
        test_data_bt = []

        for i, inst in enumerate(self.total_sample_list):
            if i in self.train_set_list[fold_no]:
                train_data.append(inst)
            elif i in self.test_set_list[fold_no]:
                test_data.append(inst)

        for i, inst in enumerate(self.total_sample_bt_list):
            if i in self.train_set_list[fold_no]:
                train_data_bt.append(inst)
            elif i in self.test_set_list[fold_no]:
                test_data_bt.append(inst)




        if self.dataset_name == 'train':
            self.instances = train_data

            if self.args.augument_data: # 加入增强数据
                self.instances.extend(train_data_bt)

            if self.args.shuffle: # 对训练数据进行shuffle
                self.instances = shuffle(self.instances, random_state=self.args.seed)
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


class Batcher2():
    def __init__(self, dataset_name, tokenizer, args):

        self.dataset_name = dataset_name
        self.fold_num = args.fold_num

        self.args = args
        self.tokenizer = tokenizer
        self.total_sample_num = 0

        self.total_sample_list = []
        self.total_sample_label = []

        self.total_sample_neg1_list = []
        self.total_sample_neg1_label = []

        self.total_sample_neg2_list = []
        self.total_sample_neg2_label = []

        self.total_sample_neg3_list = []
        self.total_sample_neg3_label = []

        self.total_sample_neg4_list = []
        self.total_sample_neg4_label = []

        self.train_set_list = []
        self.test_set_list = []

        self.kf = KFold(n_splits=self.fold_num, shuffle=True, random_state=args.seed)

        ''' 得到 qd_pairs_id '''
        self.instances = []
        self.max_seq_len = args.max_seq_len
        self.batches = []


        if not dataset_name == 'dev':
            if not os.path.exists(os.path.join(neg_preprocessed_data_dir, 'train_sample_list.pkl')):

                train_left = pd.read_csv(train_query_neg_file, sep='\t', header=None)
                train_left.columns = ['id', 'q1','q1_neg1', 'q1_neg2', 'q1_neg3', 'q1_neg4', 'ori1']
                train_right = pd.read_csv(train_reply_neg_file, sep='\t', header=None)
                train_right.columns = ['id', 'id_sub', 'q2', 'label', 'q2_neg1', 'q2_neg2', 'q2_neg3', 'q2_neg4', 'ori2']
                self.df_train = train_left.merge(train_right, how='left')  # 得到训练数据
                self.df_train['q2'] = self.df_train['q2'].fillna('你好')


                for _, instance in tqdm(self.df_train[['q1', 'q2', 'label', 'id', 'id_sub',
                                                       'q1_neg1', 'q1_neg2', 'q1_neg3', 'q1_neg4',
                                                       'q2_neg1', 'q2_neg2', 'q2_neg3', 'q2_neg4']].iterrows()):
                    qid, did, q, a, label = instance.id, instance.id_sub ,instance.q1, instance.q2, instance.label
                    q_neg1 = instance.q1_neg1
                    q_neg2 = instance.q1_neg2
                    q_neg3 = instance.q1_neg3
                    q_neg4 = instance.q1_neg4

                    a_neg1 = instance.q2_neg1
                    a_neg2 = instance.q2_neg2
                    a_neg3 = instance.q2_neg3
                    a_neg4 = instance.q2_neg4

                    ids, masks, segments = self._convert_to_transformer_inputs(q, a, self.max_seq_len)

                    assert len(ids) ==  self.max_seq_len
                    assert len(masks) ==  self.max_seq_len
                    assert len(segments) ==  self.max_seq_len
                    self.total_sample_list.append((ids, masks, segments, qid, did, label))

                    ids_neg1, masks_neg1, segments_neg1 = self._convert_to_transformer_inputs(q_neg1, a_neg1, self.max_seq_len)
                    assert len(ids_neg1) ==  self.max_seq_len
                    assert len(masks_neg1) ==  self.max_seq_len
                    assert len(segments_neg1) ==  self.max_seq_len
                    self.total_sample_neg1_list.append((ids_neg1, masks_neg1, segments_neg1, qid, did, label))

                    ids_neg2, masks_neg2, segments_neg2 = self._convert_to_transformer_inputs(q_neg2, a_neg2, self.max_seq_len)
                    assert len(ids_neg2) ==  self.max_seq_len
                    assert len(masks_neg2) ==  self.max_seq_len
                    assert len(segments_neg2) ==  self.max_seq_len
                    self.total_sample_neg2_list.append((ids_neg2, masks_neg2, segments_neg2, qid, did, label))

                    ids_neg3, masks_neg3, segments_neg3 = self._convert_to_transformer_inputs(q_neg3, a_neg3, self.max_seq_len)
                    assert len(ids_neg3) ==  self.max_seq_len
                    assert len(masks_neg3) ==  self.max_seq_len
                    assert len(segments_neg3) ==  self.max_seq_len
                    self.total_sample_neg3_list.append((ids_neg3, masks_neg3, segments_neg3, qid, did, label))

                    ids_neg4, masks_neg4, segments_neg4 = self._convert_to_transformer_inputs(q_neg4, a_neg4, self.max_seq_len)
                    assert len(ids_neg4) ==  self.max_seq_len
                    assert len(masks_neg4) ==  self.max_seq_len
                    assert len(segments_neg4) ==  self.max_seq_len
                    self.total_sample_neg4_list.append((ids_neg4, masks_neg4, segments_neg4, qid, did, label))



                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_list.pkl'), 'wb') as f:
                    pickle.dump(self.total_sample_list ,f)
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_neg1_list.pkl'), 'wb') as f:
                    pickle.dump(self.total_sample_neg1_list ,f)
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_neg2_list.pkl'), 'wb') as f:
                    pickle.dump(self.total_sample_neg2_list ,f)
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_neg3_list.pkl'), 'wb') as f:
                    pickle.dump(self.total_sample_neg3_list ,f)
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_neg4_list.pkl'), 'wb') as f:
                    pickle.dump(self.total_sample_neg4_list ,f)
                with open(os.path.join(neg_preprocessed_data_dir, 'df_train.pkl'), 'wb') as f:
                    pickle.dump(self.df_train, f)

            else:
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_list.pkl'), 'rb') as f:
                    self.total_sample_list = pickle.load(f)
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_neg1_list.pkl'), 'rb') as f:
                    self.total_sample_neg1_list  = pickle.load(f)
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_neg2_list.pkl'), 'rb') as f:
                    self.total_sample_neg2_list  = pickle.load(f)
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_neg3_list.pkl'), 'rb') as f:
                    self.total_sample_neg3_list  = pickle.load(f)
                with open(os.path.join(neg_preprocessed_data_dir, 'train_sample_neg4_list.pkl'), 'rb') as f:
                    self.total_sample_neg4_list  = pickle.load(f)
                with open(os.path.join(neg_preprocessed_data_dir, 'df_train.pkl'), 'rb') as f:
                    self.df_train = pickle.load(f)


            self.total_sample_num = len(self.total_sample_list)
            assert self.total_sample_num == len(self.total_sample_neg1_list)
            _, _, _, _, _, self.total_sample_label = zip(*self.total_sample_list)
            _, _, _, _, _, self.total_sample_neg1_label = zip(*self.total_sample_neg1_list)


            self.get_k_fold_set()
            self.set_fold(0)

        else:
            if not os.path.exists(os.path.join(neg_preprocessed_data_dir, 'test_sample_list.pkl')):

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

                with open(os.path.join(neg_preprocessed_data_dir, 'test_sample_list.pkl'), 'wb') as f:
                    pickle.dump(self.instances ,f)
                with open(os.path.join(neg_preprocessed_data_dir, 'df_test.pkl'), 'wb') as f:
                    pickle.dump(self.df_test, f)
            else:
                with open(os.path.join(neg_preprocessed_data_dir, 'test_sample_list.pkl'), 'rb') as f:
                    self.instances = pickle.load(f)
                with open(os.path.join(neg_preprocessed_data_dir, 'df_test.pkl'), 'rb') as f:
                    self.df_test = pickle.load(f)


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

        train_data_neg1 = []
        test_data_neg1 = []
        train_data_neg2 = []
        test_data_neg2 = []
        train_data_neg3 = []
        test_data_neg3 = []
        train_data_neg4 = []
        test_data_neg4 = []

        for i, inst in enumerate(self.total_sample_list):
            if i in self.train_set_list[fold_no]:
                train_data.append(inst)
            elif i in self.test_set_list[fold_no]:
                test_data.append(inst)

        for i, inst in enumerate(self.total_sample_neg1_list):
            if i in self.train_set_list[fold_no]:
                train_data_neg1.append(inst)
            elif i in self.test_set_list[fold_no]:
                test_data_neg1.append(inst)

        for i, inst in enumerate(self.total_sample_neg2_list):
            if i in self.train_set_list[fold_no]:
                train_data_neg2.append(inst)
            elif i in self.test_set_list[fold_no]:
                test_data_neg2.append(inst)

        for i, inst in enumerate(self.total_sample_neg3_list):
            if i in self.train_set_list[fold_no]:
                train_data_neg3.append(inst)
            elif i in self.test_set_list[fold_no]:
                test_data_neg3.append(inst)

        for i, inst in enumerate(self.total_sample_neg4_list):
            if i in self.train_set_list[fold_no]:
                train_data_neg4.append(inst)
            elif i in self.test_set_list[fold_no]:
                test_data_neg4.append(inst)

        if self.dataset_name == 'train':
            self.instances = train_data

            if self.args.augument_data: # 加入增强数据
                self.instances.extend(train_data_neg1)
                self.instances.extend(train_data_neg2)
                self.instances.extend(train_data_neg3)
                self.instances.extend(train_data_neg4)

            if self.args.shuffle: # 对训练数据进行shuffle
                self.instances = shuffle(self.instances, random_state=self.args.seed)
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
