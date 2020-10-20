
# coding: utf-8

# In[1]:


import logging
logging.basicConfig(level=logging.ERROR)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import *
print(tf.__version__)
from sklearn.metrics import f1_score
from transformers.tokenization_utils_base import TruncationStrategy
# from transformers.modeling_tf_bert import TFBertModel

# # 参数下载地址 https://huggingface.co/bert-base-chinese

# In[2]:



script_abs_path = os.path.dirname(__file__)
ROOT_DIR = os.path.join(script_abs_path, '../../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA = os.path.join(DATA_DIR, 'raw_data')

train_query_file = os.path.join(RAW_DATA, 'train', 'train.query.tsv')
train_reply_file = os.path.join(RAW_DATA, 'train', 'train.reply.tsv')

test_query_file = os.path.join(RAW_DATA, 'test', 'test.query.tsv')
test_reply_file = os.path.join(RAW_DATA, 'test', 'test.reply.tsv')


train_left = pd.read_csv(train_query_file,sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv(train_reply_file,sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
df_train = train_left.merge(train_right, how='left')  # 得到训练数据
df_train['q2'] = df_train['q2'].fillna('好的')
test_left = pd.read_csv(test_query_file,sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']
test_right =  pd.read_csv(test_reply_file,sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']
df_test = test_left.merge(test_right, how='left')  # 得到测试数据


# In[3]:

PATH = os.path.join(DATA_DIR, 'model_record', 'tf')
if not os.path.exists(PATH):
    os.makedirs(PATH)

BERT_PATH = os.path.join(DATA_DIR, 'pretrain_model', 'bert-base-chinese-tf')
WEIGHT_PATH = PATH
MAX_SEQUENCE_LENGTH = 100  # 最大长度 100
input_categories = ['q1','q2'] 
output_categories = 'label' # 输出类型

print('train shape =', df_train.shape)
print('test shape =', df_test.shape)


# In[16]:


def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation=truncation_strategy
            )
        
        input_ids =  inputs["input_ids"] 
        input_masks = [1] * len(input_ids) 
        input_segments = inputs["token_type_ids"] 
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length) # [max_seq]
        input_masks = input_masks + ([0] * padding_length) # [max_seq]
        input_segments = input_segments + ([0] * padding_length) # [max_seq]
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, answer, 'longest_first' , max_sequence_length)
    

    
    return [input_ids_q, input_masks_q, input_segments_q]


'''' 得到 input 数据 '''
def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q, a = instance.q1, instance.q2

        ids_q, masks_q, segments_q= _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)

        assert len(ids_q) == max_sequence_length
        assert len(masks_q) == max_sequence_length
        assert len(segments_q) == max_sequence_length

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32), # [num, max_seq] 
            np.asarray(input_masks_q, dtype=np.int32),  # [num, max_seq] 
            np.asarray(input_segments_q, dtype=np.int32)] # [num, max_seq] 

'''' 得到 label 数据 '''
def compute_output_arrays(df, columns):
    return np.asarray(df[columns]) # [num, 1]


def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(30,60): # 0.3 -- 0.6 探索 f1 值最高的 thre
        tres = i / 100
        y_pred_bin =  (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t


# In[17]:

''' 得到输入定义 '''
tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_PATH, 'bert-base-chinese-vocab.txt'))
outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# In[24]:

''' 定义模型 '''
def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    config = BertConfig.from_pretrained(os.path.join(BERT_PATH,'./bert-base-chinese-config.json'))
    config.output_hidden_states = False 
    bert_model = TFBertModel.from_pretrained(os.path.join(BERT_PATH,'./bert-base-chinese-tf_model.h5'),
                                             config=config)
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0] # [b, seq_len, hidden]
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalMaxPooling1D()(q_embedding)
    t = q_embedding[:,-1]
    e = q_embedding[:, 0]
    x = tf.keras.layers.Concatenate()([q, a, t, e])
    
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn], outputs=x)
    
    return model


# In[25]:


from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5).split(X=df_train.q2, groups=df_train.id)

valid_preds = []
test_preds = []

oof = np.zeros((len(df_train),1))
for fold, (train_idx, valid_idx) in enumerate(gkf):
    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
    train_outputs = outputs[train_idx]
    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = outputs[valid_idx]

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=[tf.keras.metrics.AUC()])
    model.fit(train_inputs, train_outputs, validation_data = (valid_inputs, valid_outputs), epochs=3, batch_size=64)
    oof_p = model.predict(valid_inputs, batch_size=512)
    oof[valid_idx] = oof_p
    valid_preds.append(oof_p)
    test_preds.append(model.predict(test_inputs, batch_size=512))
    f1,t = search_f1(valid_outputs, valid_preds[-1])
    print('validation score = ', f1)


# In[23]:


best_score, best_t = search_f1(outputs,oof)


# In[21]:


sub = np.average(test_preds, axis=0) 
sub = sub > best_t
df_test['label'] = sub.astype(int)
df_test[['id','id_sub','label']].to_csv(os.path.join( WEIGHT_PATH,'submission_beike_{}.csv'.format(best_score)),index=False, header=None,sep='\t')

