# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import collections
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import re
from sklearn import metrics


# %%
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
ANJUKE_DATA_PATH = PROJECT_ROOT_PATH/'data/anjuke'


# %%
SUB_DATA_PATHS = [x for x in ANJUKE_DATA_PATH.iterdir() if x.is_dir()]


# %%
SUB_DATA_PATHS


# %%
cities = [x.name[-2:] for x in SUB_DATA_PATHS]
cities


# %%
import json


# %%
city2samples = {}
for data_path in tqdm(SUB_DATA_PATHS):
    city = data_path.name[-2:]
    samples = []
    with open(data_path/'part_0.json', 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    samples.extend(lines)
    with open(data_path/'part_1.json', 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    samples.extend(lines)
    
    samples = [json.loads(s) for s in samples]
    city2samples[city] = samples


# %%
city2samples['bj'][0]


# %%
qid = 0
df_anjuke = []
for city, samples in tqdm(city2samples.items()):
    formatted_samples = []

    for s in samples:
        q1 = s['question']
        q2s = s['answers']

        for i, q2 in enumerate(q2s):
            formatted_samples.append([qid, q1, i, q2, 1, city])
        qid += 1
    
    df_anjuke.extend(formatted_samples)


# %%
df_anjuke = pd.DataFrame(data=df_anjuke, columns=['id', 'q1', 'id_sub', 'q2', 'label', 'city'])


# %%
df_anjuke


# %%
def get_overlap_count(a, b):
    a = list(a)
    b = list(b)
    overlap_char = set(a).intersection(b)
    return len(overlap_char)


# %%
print(df_anjuke['q2'].iloc[0], df_anjuke['q2'].iloc[1])
print(get_overlap_count(df_anjuke['q2'].iloc[0], df_anjuke['q2'].iloc[1]))


# %%
a = ['haha', 'hehe']
a.remove('haha')
print(a)


# %%
df_neg = []
for id in tqdm(df_anjuke['id'].unique()):
    df_sub = df_anjuke[df_anjuke['id'] == id]
    q1 = df_sub['q1'].unique()[0]
    num_pos_samples = df_sub.shape[0]
    city = df_sub['city'].unique()[0]

    # 随机从其他城市的答案中采样负样本，并且保证重叠字符数较低，
    # 尽量避免选到和正样本太相似的负样本
    neg_cities = list(city2samples.keys())
    neg_cities.remove(city)
    neg_city = np.random.choice(neg_cities)
    
    while True:
        neg_samples = np.random.choice(city2samples[neg_city], size=num_pos_samples*3)
        neg_samples = [x['answers'][0] for x in neg_samples]
        
        flags = []
        for pos in df_sub['q2']:
            for neg in neg_samples:
                flags.append(get_overlap_count(pos, neg))
        flags = np.asarray(flags)
        if flags.mean() < 10:
            break

    id_sub_start = max(df_sub['id_sub']) + 1
    for i, q2 in enumerate(neg_samples):
        df_neg.append([id, q1, id_sub_start + i, q2, 0, neg_city])



# %%
id2neg_samples[0]


# %%
id2neg_samples = {}
for row in df_anjuke['id', 'id_sub', 'city'].intertuples(index=False):
    neg_samples = np.random.choice(city2samples[neg_city], size=3)


# %%
train_query = pd.read_csv(RAW_DATA_PATH/'./train/train.query.tsv',sep='\t',header=None)
train_query.columns=['id','q1']
train_reply = pd.read_csv(RAW_DATA_PATH/'./train/train.reply.tsv',sep='\t',header=None)
train_reply.columns=['id','id_sub','q2','label']
train_df = train_query.merge(train_reply, how='left')
train_df['q2'] = train_df['q2'].fillna('好的')
test_query = pd.read_csv(RAW_DATA_PATH/'./test/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_query.columns = ['id','q1']
test_reply =  pd.read_csv(RAW_DATA_PATH/'./test/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_reply.columns=['id','id_sub','q2']
test_df = test_query.merge(test_reply, how='left')


# %%
train_query.head()


# %%
train_df.head()


# %%
print(f'训练集问题数量：{len(train_query)}，答案数量：{len(train_reply)}')
print(f'测试集问题数量：{len(test_query)}，答案数量：{len(test_reply)}')


# %%
num_label = [0, 0]
for label in train_df['label']:
    num_label[label] += 1
print(f'训练集中正样本数量：{num_label[1]}，负样本数量：{num_label[0]}')


# %%
# func = lambda x: bool(re.search('[是]', str(x)))
# q1_bool = train_df['q1'].apply(func=func)
# q2_bool = train_df['q2'].apply(func=func)


# %%
# possible_df = train_df[train_df['q2'].apply(lambda x: bool(re.search('[是的]', str(x))))]
# # possible_df = train_df[q1_bool & q2_bool] 
# possible_df.to_csv(RAW_DATA_PATH/'./train/train.possible.tsv',sep='\t')


# %%
q_length = train_query['q1'].apply(func=lambda x: len(x)).sort_values()
a_length = train_reply['q2'].apply(func=lambda x: len(str(x))).sort_values()


# %%
print(f'#question: {len(train_query)}, #answer: {len(train_reply)}')
print('问题和回答的长度分布：')
plt.subplot(121)
plt.hist(q_length[:-50], density=True, cumulative=True, bins=20)
plt.subplot(122)
plt.hist(a_length.sort_values()[:-500], density=True, cumulative=True, bins=20)
plt.show()


# %%
a_pos_length = train_reply['q2'][train_reply['label'] == 1].apply(func=lambda x: len(str(x))).sort_values()
plt.hist(a_pos_length.sort_values()[:-100], density=True, cumulative=True, bins=20)
plt.show()


# %%
answers = train_df[train_df['id'] == 100]
answers


# %%
answers['label']


# %%
pos_ratioes = {id: [0, 0] for id in train_query['id']}
for row in tqdm(train_df.itertuples()):
    pos_ratioes[row.id][row.label] += 1
for k, v in pos_ratioes.items():
    pos_ratioes[k] = v[1] / sum(v)


# %%
pos_neg_num = {id: [0, 0] for id in train_query['id']}
for row in tqdm(train_df.itertuples()):
    pos_neg_num[row.id][row.label] += 1


# %%
num_no_pos = 0
for v in pos_ratioes.values():
    if v == 0.0:
        num_no_pos += 1
print(f'Ratio of queries without positive answers: {num_no_pos/len(pos_ratioes)}.')
plt.hist(pos_ratioes.values(), bins=30, density=True, cumulative=True)
plt.show()


# %%
# train_df[train_df['label'] == 1].to_csv(RAW_DATA_PATH/'./train/train.pos.tsv',sep='\t',index=False)


# %%
train_query_bt = pd.read_csv(RAW_DATA_PATH/'./train/train.query.bt.tsv',sep='\t',header=None, names=['id','q1','q1_bt'])
train_query_bt


# %%
train_query_bt['q1'] = train_query_bt['q1_bt']
train_query_bt = train_query_bt.drop('q1_bt', axis='columns')


# %%
train_reply_bt = pd.read_csv(RAW_DATA_PATH/'./train/train.reply.bt.tsv',sep='\t',header=None, names=['id','id_sub', 'q2', 'label','q2_bt'])
train_reply_bt


# %%
train_reply_bt['q2'] = train_reply_bt['q2_bt']
train_reply_bt = train_reply_bt.drop('q2_bt', axis='columns')


# %%
train_df_bt = pd.merge(train_query_bt, train_reply_bt, how='left')
train_df_bt


# %%
# 只增强正样本
# train_df_bt = train_df_bt[train_df_bt['label'] == 1]
# train_df_bt


# %%
train_df_bt['q2'] = train_df_bt['q2'].fillna('好的')


# %%
# 将增强的样本和原训练集合并
train_df_ex = pd.concat([train_df, train_df_bt])
train_df_ex


# %%
train_id_set = set([f'{x[0]},{x[1]}' for x in train_df[train_df['label'] == 1][['id', 'id_sub']].to_numpy()])
mask = train_df_bt[['id', 'id_sub']].apply(lambda x: f'{x["id"]},{x["id_sub"]}' in train_id_set, axis=1)
mask


# %%
train_df_bt[mask]


# %%
train_df_ex.to_csv(RAW_DATA_PATH/'train/train_ex.tsv', sep='\t', index=False, header=None)


# %%
# train_reply_bt[train_reply_bt['label'] == 1].to_csv(RAW_DATA_PATH/'./train/train.reply.bt.pos.tsv',sep='\t',index=False)


# %%
train_reply_aug = pd.read_csv(RAW_DATA_PATH/'./train/train.reply.aug4.tsv',sep='\t',header=None, names=['id','id_sub', 'q2', 'label', 'q2_aug_0', 'q2_aug_1', 'q2_aug_2', 'q2_aug_3', '_']).drop('_', axis=1)
train_reply_aug


# %%
train_query_aug = pd.read_csv(RAW_DATA_PATH/'./train/train.query.aug4.tsv',sep='\t',header=None, names=['id','q1','q1_aug_0', 'q1_aug_1', 'q1_aug_2', 'q1_aug_3', '_']).drop(['_'], axis=1)
train_query_aug


# %%
from collections import defaultdict


# %%
train_reply_aug_dict = defaultdict(list)
for row in tqdm(train_reply_aug.itertuples(index=False)):
    idx = row.id
    idx_sub = row.id_sub
    label = row.label
    samples = [row.q2] + list(row[4:])
    num_samples = len(samples)

    train_reply_aug_dict['id'].extend([idx] * num_samples)
    train_reply_aug_dict['id_sub'].extend([idx_sub] * num_samples)
    train_reply_aug_dict['label'].extend([label] * num_samples)
    train_reply_aug_dict['q2'].extend(samples)


# %%
train_reply_aug = pd.DataFrame(data=train_reply_aug_dict)
train_df_aug = pd.merge(train_query, train_reply_aug, how='left')
train_df_aug = train_df_aug[['id', 'q1', 'id_sub', 'q2', 'label']]


# %%
train_df_aug


# %%
train_df_aug.to_csv(RAW_DATA_PATH/'train/train_aug_answer_only.tsv', sep='\t', index=False, header=None)


# %%
# from nltk.util import ngrams
# ns = list(range(1, 4))
# def get_ngram_vector(grams, vocab):
#     v = np.zeros(shape=[len(vocab)])
#     for g in grams: 
#         if g in vocab:
#             v[vocab[g]] += 1
#     return v


# %%
# vocab = {'n': 0, 'haha': 1}
# grams = ['n', 'haha', 'n', '', 'haha', 2342]
# get_ngram_vector(grams, vocab)


# %%
# def get_ngram(sequence, n):
#     return [''.join(x) for x in ngrams(sequence=sequence, n=n)]


# %%
# get_ngram('nihao', 3)


# %%
# # from sklearn.metrics.pairwise import cosine_similarity
# from numpy import dot
# from numpy.linalg import norm
# def cos_sim(a, b):
    # return dot(a, b)/(norm(a)*norm(b))


# %%
# for i, sample in tqdm(enumerate(train_query_aug.itertuples(index=False))):
#     org_q = sample[1]
#     org_grams = []
#     for n in ns:
#         org_grams.extend(get_ngram(org_q, n))
#     vocab = dict(zip(set(org_grams), range(len(org_grams))))
#     # print(vocab)
#     org_v = get_ngram_vector(org_grams, vocab)

#     aug_qs = sample[2:]
#     aug_vs = np.zeros(shape=[4, len(vocab)])
#     for i, aug_q in enumerate(aug_qs):
#         aug_grams = []
#         for n in ns:
#             aug_grams.extend(get_ngram(aug_q, n))
#         aug_v = get_ngram_vector(aug_grams, vocab)
#         aug_vs[i] = aug_v

#     sims = [cos_sim(org_v, v) for v in aug_vs]
#     print(sims)
#     print(org_q)
#     print(aug_qs)


# %%
import transformers


# %%
tokenizer = transformers.BertTokenizer.from_pretrained('../pretrain_model/bert-base-chinese/vocab.txt')


# %%
tokenizer.encode_plus('你好啊小老弟')


# %%
tokenizer.encode_plus(list('你好啊小老弟'))


# %%
tokenizer('你好啊小老弟')


# %%



