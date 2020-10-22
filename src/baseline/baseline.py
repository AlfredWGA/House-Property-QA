# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import logging
logging.basicConfig(level=logging.ERROR)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# import tensorflow as tf
# import tensorflow.keras.backend as K
import os
# from transformers import *
# print(tf.__version__)
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from house_dataset import HouseDataset
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, TFBertModel, BertConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 设定显卡
assert torch.cuda.is_available()
device = torch.device("cuda:3")

# null.tpl [markdown]
# # 参数下载地址 https://huggingface.co/bert-base-chinese

PROJECT_ROOT_PATH = (Path(__file__).parent.parent.parent).resolve()
RAW_DATA_PATH = PROJECT_ROOT_PATH/'data/raw_data'

# %%
train_left = pd.read_csv(RAW_DATA_PATH/'./train/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv(RAW_DATA_PATH/'./train/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
df_train = train_left.merge(train_right, how='left')
df_train['q2'] = df_train['q2'].fillna('好的')
test_left = pd.read_csv(RAW_DATA_PATH/'./test/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']
test_right =  pd.read_csv(RAW_DATA_PATH/'./test/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']
df_test = test_left.merge(test_right, how='left')


# %%
# PATH = './'
# BERT_PATH = './data/pretrain_model'
# WEIGHT_PATH = './'
MAX_SEQUENCE_LENGTH = 100
# CHECKPOINT_PATH = PROJECT_ROOT_PATH'./ckpt'
PRETRAIN_MODEL_PATH = PROJECT_ROOT_PATH/'./data/pretrain_model'
input_categories = ['q1','q2']
output_categories = 'label'

print('train shape =', df_train.shape)
print('test shape =', df_test.shape)


# %%
def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):
        # inputs = tokenizer.encode_plus(str1, str2,
        #     add_special_tokens=True,
        #     max_length=length,
        #     truncation_strategy=truncation_strategy,
        #     truncation=True,
        #     )
        
        # input_ids =  inputs["input_ids"]
        # input_masks = [1] * len(input_ids)
        # input_segments = inputs["token_type_ids"]
        # padding_length = length - len(input_ids)
        # padding_id = tokenizer.pad_token_id
        # input_ids = input_ids + ([padding_id] * padding_length)
        # input_masks = input_masks + ([0] * padding_length)
        # input_segments = input_segments + ([0] * padding_length)

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            # truncation_strategy='longest_first',
            truncation='longest_first',
            padding='max_length'
            )
        # print(inputs.keys())
        input_ids = inputs["input_ids"]
        input_segments = inputs["token_type_ids"]
        input_masks = inputs['attention_mask']

        return [input_ids, input_masks, input_segments]


    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, answer, 'longest_first', max_sequence_length)
    

    
    return [input_ids_q, input_masks_q, input_segments_q]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q, a = instance.q1, instance.q2

        ids_q, masks_q, segments_q = _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(30,60):
        tres = i / 100
        y_pred_bin =  (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t


# %%

tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL_PATH/'bert-base-chinese'/'vocab.txt')

inputs_path = Path(RAW_DATA_PATH/'./train/inputs.npy')
outputs_path = Path(RAW_DATA_PATH/'./train/outputs.npy')
test_inputs_path = Path(RAW_DATA_PATH/'./test/test_inputs.npy')

# 保存处理好的数据集
if not outputs_path.exists():
    outputs = compute_output_arrays(df_train, output_categories)
    np.save(outputs_path, outputs)
else:
    outputs = np.load(outputs_path, allow_pickle=True)

if not inputs_path.exists():
    inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    np.save(inputs_path, inputs)
else:
    inputs = np.load(inputs_path, allow_pickle=True)

if not test_inputs_path.exists():
    test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    np.save(test_inputs_path, test_inputs)
else:
    test_inputs = np.load(test_inputs_path, allow_pickle=True)    


# Pytorch版BERT模型
class BertForHouseQA(nn.Module):
    def __init__(self):
        super(BertForHouseQA, self).__init__()
        self.bert = BertModel.from_pretrained(os.path.join(PRETRAIN_MODEL_PATH, 'bert-base-chinese'), 
                    output_hidden_states=False, 
                    output_attentions=False)
                    
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(4*self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):
        hidden_state = self.bert(input_ids=x[:, 0, :], attention_mask=x[:, 1, :], token_type_ids=x[:, 2, :])[0]
        q, _ = torch.max(hidden_state, dim=1) # [b, hid]
        a = torch.mean(hidden_state, dim=1) # [b, hid]
        t = hidden_state[:, -1] # [b, hid]
        e = hidden_state[:, 0] # [b, hid]

        feat = torch.cat([q, a, t, e], dim=-1) # [b, 4*hid]

        feat = self.dropout(feat)
        logit = self.fc(feat)
        pre = self.sigmoid(logit)
        return pre
    

def train_pytorch(**kwargs):
    inputs = kwargs['inputs']
    outputs = kwargs['outputs']
    test_inputs = kwargs['test_inputs']
    
    gkf = GroupKFold(n_splits=kwargs['n_splits']).split(X=df_train.q2, groups=df_train.id)
    
    oof = np.zeros((len(df_train),1))
    test_preds = []

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        logger.info(f'Fold No. {fold}')
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]
        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]

        train_set = HouseDataset(train_inputs, train_outputs)
        valid_set = HouseDataset(valid_inputs, valid_outputs)
        test_set = HouseDataset(test_inputs, np.zeros_like(test_inputs[0])) # 测试集没有标签

        logger.info('Train set size: {}, valid set size {}'.format(
            len(train_set), len(valid_set)))

        train_loader = DataLoader(train_set,
                                batch_size=kwargs['batch_size'],
                                shuffle=True)

        valid_loader = DataLoader(valid_set,
                                batch_size=512)

        test_loader = DataLoader(test_set,
                                batch_size=512)

        model = BertForHouseQA().cuda(device)

        # List all modules inside the model.
        # logger.info('Model modules:')
        # for i, m in enumerate(model.named_children()):
        #     logger.info('{} -> {}'.format(i, m))

        # # Get the number of total parameters.
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel()
        #                     for p in model.parameters() if p.requires_grad)

        # logger.info("Total params: {:,}".format(total_params))
        # logger.info("Trainable params: {:,}".format(trainable_params))

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        mode='min',
        #                                                        patience=8,
        #                                                        verbose=True
        #                                                        )

        for epoch in range(kwargs['epoch']):
            # =======================Training===========================
            # Set model to train mode.
            model.train()
            steps = int(np.ceil(len(train_set) / kwargs['batch_size']))
            pbar = tqdm(desc='Epoch {}, loss {}'.format(epoch, 'NAN'),
                        total=steps)
            for i, sample in enumerate(train_loader):
                x, y = sample[0].cuda(device).long(), sample[1].cuda(device).float()
                optimizer.zero_grad()

                model_outputs = model(x)
                loss = criterion(model_outputs, y.unsqueeze(-1))

                loss.backward()
                optimizer.step()
                pbar.set_description(
                    'Epoch {}, training loss {:.5f}'.format(epoch, loss.item()))
                pbar.update()
            pbar.close()
            # =========================================================
            # =======================Validation========================
            # Set model to evaluation mode.
            model.eval()
            with torch.no_grad():
                # Validation step
                valid_loss = []
                valid_pred = []
                valid_true = []
                for i, sample in enumerate(valid_loader):
                    y_true_local = sample[1].numpy()
                    x, y_true = sample[0].cuda(
                        device).long(), sample[1].cuda(device).float()

                    model_outputs = model(x)
                    loss = criterion(model_outputs, y_true.unsqueeze(-1)).cpu().detach().item()
                    # y_pred = outputs.argmax(dim=1).cpu().numpy()
                    y_pred = model_outputs.cpu().detach().numpy()
                    valid_loss.append(loss)
                    valid_pred.append(y_pred)
                    valid_true.append(y_true_local)
            valid_loss = np.asarray(valid_loss).mean()
            valid_pred = np.concatenate(valid_pred, axis=0)
            valid_true = np.concatenate(valid_true, axis=0)
            valid_auc = roc_auc_score(valid_true, valid_pred)
            
            # Apply ReduceLROnPlateau to the lr.
            # scheduler.step(valid_loss)

            logger.info('Epoch {}, valid loss {:.5f}, valid auc {:.4f}'.format(
                epoch, valid_loss, valid_auc))

            # 保存每个验证折的预测值，用作最后整个训练集的f1评估
            oof[valid_idx] = valid_pred
            valid_f1, _ = search_f1(valid_outputs, valid_pred)  # 寻找最佳分类阈值和f1 score
            print('Valid f1 score = ', valid_f1)
            # ==========================================================
            # =======================Prediction========================
            # Set model to evaluation mode.
            model.eval()
            with torch.no_grad():
                # Prediction step
                test_pred = []
                steps = int(np.ceil(len(test_set) / 512))
                for i, sample in tqdm(enumerate(test_loader), desc='Predicting', total=steps):
                    y_true_local = sample[1].numpy()
                    x, y_true = sample[0].cuda(
                        device).long(), sample[1].cuda(device).float()

                    model_outputs = model(x)
                    # loss = criterion(model_outputs, y_true.unsqueeze(-1)).cpu().detach().item()
                    # y_pred = outputs.argmax(dim=1).cpu().numpy()
                    y_pred = model_outputs.cpu().detach().numpy()
                    test_pred.append(y_pred)
            test_pred = np.concatenate(test_pred, axis=0)
            test_preds.append(test_pred)
            # ==========================================================
            # Save the model at the end of every epoch.
            # torch.save({
            #     'model_name': 'BertForHouseQA',
            #     'epoch': epoch,
            #     'loss': loss,
            #     'valid_f1': valid_f1,
            #     'model_state_dict':  model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     # 'scheduler_state_dict': scheduler.state_dict()
            # }, f=os.path.join(CHECKPOINT_PATH, 'ckpt.pt'))

    return oof, test_preds

oof, test_preds = train_pytorch(batch_size=128, epoch=5, n_splits=5, inputs=inputs, outputs=outputs, test_inputs=test_inputs)

# %%
# def create_model():
#     q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
#     q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
#     q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
#     config = BertConfig.from_pretrained('./bert-base-chinese-config.json') 
#     config.output_hidden_states = False 
#     bert_model = TFBertModel.from_pretrained('./bert-base-chinese-tf_model.h5', 
#                                              config=config)
#     q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
#     q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
#     a = tf.keras.layers.GlobalMaxPooling1D()(q_embedding)
#     t = q_embedding[:,-1]
#     e = q_embedding[:, 0]
#     x = tf.keras.layers.Concatenate()([q, a, t, e])
    
#     x = tf.keras.layers.Dropout(0.5)(x)
#     x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#     model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn], outputs=x)
    
#     return model


# %%

# gkf = GroupKFold(n_splits=5).split(X=df_train.q2, groups=df_train.id)

# valid_preds = []
# test_preds = []

# oof = np.zeros((len(df_train),1))
# for fold, (train_idx, valid_idx) in enumerate(gkf):
#     train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
#     train_outputs = outputs[train_idx]
#     valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
#     valid_outputs = outputs[valid_idx]

#     K.clear_session()
#     model = create_model()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
#     model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=[tf.keras.metrics.AUC()])
#     model.fit(train_inputs, train_outputs, validation_data = (valid_inputs, valid_outputs), epochs=3, batch_size=64)
#     oof_p = model.predict(valid_inputs, batch_size=512)
#     oof[valid_idx] = oof_p
#     valid_preds.append(oof_p)
#     test_preds.append(model.predict(test_inputs, batch_size=512))
#     f1,t = search_f1(valid_outputs, valid_preds[-1])
#     print('validation score = ', f1)


# %%
# 整个训练集的f1评估，寻找最佳的分类阈值
best_score, best_t = search_f1(outputs, oof)


# %%
sub = np.average(test_preds, axis=0) 
sub = sub > best_t  # 用该分类阈值来输出测试集
df_test['label'] = sub.astype(int)
df_test[['id','id_sub','label']].to_csv('submission_beike_{}.csv'.format(best_score),index=False, header=None,sep='\t')


# %%



