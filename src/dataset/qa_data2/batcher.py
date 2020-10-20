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
        self.qd_pairs_id = []
        self.query = {}
        self.doc = {}

        if not dataset_name == 'dev':
            qd_pairs = []
            with open(train_query_file, 'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    qid = line[0]
                    query = line[1]
                    self.query[qid] = query

            with open(train_reply_file, 'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    qid = line[0]
                    did = line[1]
                    doc = line[2]
                    label = int(line[3])
                    self.doc[did] = doc
                    qd_pairs.append((qid, did, label))

            train_set, test_set = train_test_split(qd_pairs, test_size=0.2, random_state=42)
            if dataset_name == 'train':
                self.qd_pairs_id = list(train_set)
            else:
                self.qd_pairs_id = list(test_set)
        else:
            with open(test_query_file, 'r', encoding='gbk') as f:
                for line in f:
                    line = line.strip().split('\t')
                    qid = line[0]
                    query = line[1]
                    self.query[qid] = query

            with open(test_reply_file, 'r', encoding='gbk') as f:
                for line in f:
                    line = line.strip().split('\t')
                    qid = line[0]
                    did = line[1]
                    doc = line[2]
                    self.doc[did] = doc
                    self.qd_pairs_id.append((qid, did, 1))



    ''' 得到该批数据所有的 qid did 和 label '''
    def get_all(self):

        X_qid, X_did, Y = zip(*self.qd_pairs_id)
        X = {'id_left': np.array(X_qid),
             'id_right': np.array(X_did)}
        Y = np.array(Y)

        return X, Y

    ''' 得到query内容 '''
    def get_query(self, qid):
        q_str = self.query[qid]
        return q_str

    ''' 得到doc内容 '''
    def get_doc(self, did):
        d_str = self.doc[did]

        return d_str



if __name__ == '__main__':
    batcher = Batcher(dataset_name='test', tokenizer=None, args=None)
