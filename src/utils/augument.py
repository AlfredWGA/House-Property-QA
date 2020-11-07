# coding=utf-8

import os
import pandas as pd
from eda import *
from tqdm import tqdm

script_abs_path = os.path.dirname(__file__)
ROOT_DIR = os.path.join(script_abs_path, '../../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA = os.path.join(DATA_DIR, 'raw_data')

train_query_file = os.path.join(RAW_DATA, 'train', 'train.query.tsv')
train_reply_file = os.path.join(RAW_DATA, 'train', 'train.reply.tsv')

test_query_file = os.path.join(RAW_DATA, 'test', 'test.query.tsv')
test_reply_file = os.path.join(RAW_DATA, 'test', 'test.reply.tsv')

train_dir = os.path.join(RAW_DATA, 'train')
test_dir = os.path.join(RAW_DATA, 'test')

def get_eda(sentence, alpha=0.1, num_aug=4):
    aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
    aug_sentences = [x.replace(' ', '') for x in aug_sentences]
    return aug_sentences

if __name__ == '__main__':
    sents = get_eda('这套房源价格还有优惠空间吗？')

    print(sents)

    # train_left = pd.read_csv(train_query_file, sep='\t', header=None)
    # train_left.columns = ['id', 'q1']
    #
    #
    # train_query_btrans_file =  train_query_file.replace('.tsv', '.aug4.tsv')
    #
    # for _, instance in tqdm(train_left[['id', 'q1']].iterrows()):
    #     q = instance.q1
    #     id = instance.id
    #     q_negs = get_eda(q)
    #     q_negs = '\t'.join(q_negs)
    #     print(q, ' ', q_negs)
    #
    #     with open(train_query_btrans_file, 'a') as f:
    #         line = '\t'.join([str(id), str(q), str(q_negs)]) + '\n'
    #         f.write(line)


    # train_right = pd.read_csv(train_reply_file, sep='\t', header=None)
    # train_right.columns = ['id', 'id_sub', 'q2', 'label']
    # train_right['q2'] = train_right['q2'].fillna('你好')
    #
    # train_reply_btrans_file =  train_reply_file.replace('.tsv', '.aug4.tsv')
    # q_bt = []
    # for _, instance in tqdm(train_right[['id', 'id_sub', 'q2', 'label']].iterrows()):
    #     label = instance.label
    #     id_sub = instance.id_sub
    #     id = instance.id
    #     q = instance.q2
    #     if q == None or q == '':
    #         q = '你好'
    #     q_negs = get_eda(q)
    #     q_negs = '\t'.join(q_negs)
    #     print(q, ' ', q_negs)
    #     with open(train_reply_btrans_file, 'a') as f:
    #         line = '\t'.join([str(id), str(id_sub), str(q), str(label), str(q_negs)]) + '\n'
    #         f.write(line)

