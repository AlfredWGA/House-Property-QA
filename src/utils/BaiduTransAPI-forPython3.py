#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json
import time
import os
import pandas as pd
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


# preprocessed_data_dir = os.path.join(DATA_DIR, 'preprocessed_data')


def translate(q, fromLang='zh', toLang='en'):

    appid = '20201024000597247'  # 填写你的appid
    secretKey = '47oaTX5Eh8OxRLf92eFR'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = fromLang  # 原文语种
    toLang = toLang  # 译文语种
    salt = random.randint(32768, 65536)
    q = q
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        res = result['trans_result'][0]['dst']

    except Exception as e:
        print(e)
        res = None
    finally:
        if httpClient:
            httpClient.close()
    time.sleep(1)

    return res

def back_translate(q):
    q = translate(q, fromLang='zh', toLang='jp')
    if q == None or q == '':
        q = '好的'
    q = translate(q, fromLang='jp', toLang='zh')
    return q





if __name__ == '__main__':
    # res = back_translate('总房款多少')
    # print(res)

    # train_left = pd.read_csv(train_query_file, sep='\t', header=None)
    # train_left.columns = ['id', 'q1']
    #
    #
    # train_query_btrans_file =  train_query_file.replace('.tsv', '.bt.jp.tsv')
    #
    # for _, instance in tqdm(train_left[['id', 'q1']].iterrows()):
    #     q = instance.q1
    #     id = instance.id
    #     q_tran = back_translate(q)
    #     print(q, ' ', q_tran)
    #
    #     with open(train_query_btrans_file, 'a') as f:
    #         line = '\t'.join([str(id), str(q), str(q_tran)]) + '\n'
    #         f.write(line)


    train_right = pd.read_csv(train_reply_file, sep='\t', header=None)
    train_right.columns = ['id', 'id_sub', 'q2', 'label']
    train_right['q2'] = train_right['q2'].fillna('你好')

    train_reply_btrans_file =  train_reply_file.replace('.tsv', '.bt.jp.tsv')
    q_bt = []
    for _, instance in tqdm(train_right[['id', 'id_sub', 'q2', 'label']].iterrows()):
        label = instance.label
        id_sub = instance.id_sub
        id = instance.id
        q = instance.q2
        if q == None or q == '':
            q = '你好'
        q_tran = back_translate(q)
        # q_bt.append(q_tran)
        print(q, ' ', q_tran)
        with open(train_reply_btrans_file, 'a') as f:
            line = '\t'.join([str(id), str(id_sub), str(q), str(label), str(q_tran)]) + '\n'
            f.write(line)




