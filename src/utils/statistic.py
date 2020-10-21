# coding=utf-8

import logging
import os
import csv
import gzip
import json

from utils.tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("Batcher")
logger.setLevel(logging.DEBUG)



script_abs_path = os.path.dirname(__file__)
ROOT_DIR = os.path.join(script_abs_path, '../../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA = os.path.join(DATA_DIR, 'raw_data')

train_query_file = os.path.join(RAW_DATA, 'train', 'train.query.tsv')
train_reply_file = os.path.join(RAW_DATA, 'train', 'train.reply.tsv')

test_query_file = os.path.join(RAW_DATA, 'test', 'test.query.tsv')
test_reply_file = os.path.join(RAW_DATA, 'test', 'test.reply.tsv')

from sklearn.model_selection import KFold

if __name__ == '__main__':
    data = list(range(10))
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(data):
        print('Train', train_index)
        print('Test', test_index)\

    print('------------------------------------------------')
    for train_index, test_index in kf.split(data):
        print('Train', train_index)
        print('Test', test_index)


    # qd_pairs = []
    # query = {}
    # doc = {}
    # tokenizer = Tokenizer(vocab_dir='bert-base-chinese')
    # len_doc = 0
    # doc_num = 0
    # max_doc_len = 0
    # doc_len_list = []
    # with open(train_reply_file, 'r') as f:
    #     for i, line in enumerate(f):
    #         line = line.strip().split('\t')
    #         qid = line[0]
    #         did = line[1]
    #         doc_str = line[2]
    #         d_len = len(tokenizer.cut(doc_str))
    #         doc_len_list.append(d_len)
    #         if d_len > max_doc_len:
    #             max_doc_len = d_len
    #         len_doc += d_len
    #         doc_num += 1
    # doc_len_list = sorted(doc_len_list, reverse=True)
    # logger.info(doc_len_list[:100])
    # logger.info('train_reply avg len %f , max len %d' % ((len_doc/doc_num), max_doc_len))
    #
    #
    # len_query = 0
    # query_num = 0
    # max_q_len = 0
    # q_len_list = []
    # with open(train_query_file, 'r') as f:
    #     for i, line in enumerate(f):
    #         line = line.strip().split('\t')
    #         qid = line[0]
    #         query = line[1]
    #         q_len = len(tokenizer.cut(query))
    #         q_len_list.append(q_len)
    #         if q_len > max_q_len:
    #             max_q_len = q_len
    #         len_query += q_len
    #         query_num += 1
    # q_len_list = sorted(q_len_list, reverse=True)
    # logger.info(q_len_list[:100])
    # logger.info('train_query avg len %f , max len %d' % ((len_query / query_num), max_q_len))
    #
    #
    # len_query = 0
    # query_num = 0
    # max_q_len = 0
    # q_len_list = []
    # with open(test_query_file, 'r', encoding='gbk') as f:
    #     for i, line in enumerate(f):
    #         line = line.strip().split('\t')
    #         qid = line[0]
    #         query = line[1]
    #         q_len = len(tokenizer.cut(query))
    #         q_len_list.append(q_len)
    #         if q_len > max_q_len:
    #             max_q_len = q_len
    #         len_query += q_len
    #         query_num += 1
    # q_len_list = sorted(q_len_list, reverse=True)
    # logger.info(q_len_list[:100])
    # logger.info('test_query avg len %f , max len %d' % ((len_query / query_num), max_q_len))
    #
    # len_doc = 0
    # doc_num = 0
    # max_doc_len = 0
    # doc_len_list = []
    # with open(test_reply_file, 'r', encoding='gbk') as f:
    #     for i, line in enumerate(f):
    #         line = line.strip().split('\t')
    #         qid = line[0]
    #         did = line[1]
    #         doc_str = line[2]
    #         d_len = len(tokenizer.cut(doc_str))
    #         doc_len_list.append(d_len)
    #         if d_len > max_doc_len:
    #             max_doc_len = d_len
    #         doc_num += 1
    #         len_doc += d_len
    # doc_len_list = sorted(doc_len_list, reverse=True)
    # logger.info(doc_len_list[:100])
    # logger.info('test_reply avg len %f , max len %d' % ((len_doc/doc_num), max_doc_len))








