# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from collections import defaultdict
import logging
import re
from typing import Pattern
from numpy.core.fromnumeric import alltrue
from sklearn.utils import shuffle
from transformers.file_utils import ModelOutput
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# import tensorflow as tf
# import tensorflow.keras.backend as K
import os
# from transformers import *
# print(tf.__version__)
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_fscore_support
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit, StratifiedKFold
from house_dataset import HouseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, TFBertModel, BertConfig
import random
from early_stopping import EarlyStopping


# formater = logging.Formatter(
    # '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# logger.setFormatter(formater)
# logger.setLevel(logging.INFO)

assert torch.cuda.is_available()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set random seed
# RANDOM_SEED = 89549
# setup_seed(RANDOM_SEED)

PROJECT_ROOT_PATH = (Path(__file__)).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT_PATH / "data"
RAW_DATA_PATH = DATA_PATH/'raw_data'

# %%
train_left = pd.read_csv(RAW_DATA_PATH/'./train/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv(RAW_DATA_PATH/'./train/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
df_train = train_left.merge(train_right, how='left')
df_train['q2'] = df_train['q2'].fillna('好的')

# df_train_ex = pd.read_csv(RAW_DATA_PATH/'./train/train_ex.tsv', sep='\t', header=None, names=['id', 'q1', 'id_sub', 'q2', 'label'])

# train_reply_bt = pd.read_csv(RAW_DATA_PATH/'./train/train.reply.bt.tsv', sep='\t',
        # header=None, names=['id','id_sub', 'q2', 'label','q2_bt'])
# train_reply_bt_pos = train_reply_bt[train_reply_bt['label'] == 1]
# train_reply_bt_pos['q2'] = train_reply_bt_pos['q2_bt']
# train_reply_bt_pos = train_reply_bt_pos.drop('q2_bt', axis='columns')

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
# MODEL_NAME = "bert-base-chinese"
MODEL_NAME = 'chinese_wwm_ext_pytorch'
TIME_STR = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
CHECKPOINT_PATH = DATA_PATH / f"model_record/{MODEL_NAME}/{TIME_STR}"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.mkdir(parents=True)
PRETRAIN_MODEL_PATH = PROJECT_ROOT_PATH/'./data/pretrain_model'
input_categories = ['q1','q2']
output_categories = 'label'

print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formater = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Print logs to the terminal.
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formater)
# # Save logs to file.
log_path = CHECKPOINT_PATH / 'train.log'
file_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
file_handler.setFormatter(formater)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


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

tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL_PATH/MODEL_NAME/'vocab.txt')

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
        # self.bert = BertModel.from_pretrained(os.path.join(PRETRAIN_MODEL_PATH, 'bert-base-chinese'), 
        self.bert = BertModel.from_pretrained(os.path.join(PRETRAIN_MODEL_PATH, MODEL_NAME), 
                    output_hidden_states=False, 
                    output_attentions=False)
                    
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(self.bert.config.hidden_size, 2)
        self.fc = nn.Linear(3*self.bert.config.hidden_size, 2)
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        # hidden_state = self.bert(input_ids=x[:, 0, :], attention_mask=x[:, 1, :], token_type_ids=x[:, 2, :])[0]
        # q, _ = torch.max(hidden_state, dim=1) # [b, hid]
        # a = torch.mean(hidden_state, dim=1) # [b, hid]
        # t = hidden_state[:, -1] # [b, hid]
        # e = hidden_state[:, 0] # [b, hid]

        # feat = torch.cat([q, a, t, e], dim=-1) # [b, 4*hid]

        # feat = self.dropout(feat)
        # logit = self.fc(feat)
        # pre = self.sigmoid(logit)
        hidden_state = self.bert(
            input_ids=x[:, 0, :], attention_mask=x[:, 1, :], token_type_ids=x[:, 2, :]
        )[0]    # [b, seq, hid]

        feat_cls = hidden_state[:, 0, :]    # [CLS]
        feat_mean = torch.mean(hidden_state, dim=1) # [b, hid]
        feat_max, _ = torch.max(hidden_state, dim=1) # [b, hid]
        feat = torch.cat([feat_cls, feat_mean, feat_max], dim=-1) # [b, 3*hid]

        feat = self.dropout(feat)
        # logit = self.fc(self.relu(feat))
        # logit = self.fc(self.tanh(feat))
        logit = self.fc(feat)

        return logit


def train_pytorch(**kwargs):
    inputs = kwargs['inputs']
    outputs = kwargs['outputs']
    test_inputs = kwargs['test_inputs']
    
    gkf = GroupKFold(n_splits=kwargs['n_splits']).split(X=df_train.q2, groups=df_train.id)

    # sss = StratifiedShuffleSplit(n_splits=kwargs['n_splits'], test_size=0.2, random_state=RANDOM_SEED).split(X=df_train.q2, 
            # y=df_train.label)
    # skf = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=RANDOM_SEED).split(X=df_train.q2, y=outputs)

    # oof = np.zeros((len(df_train),1))
    test_preds = []
    all_pred = np.zeros(shape=(len(df_train), 2))
    all_true = np.zeros(shape=(len(df_train)))
    best_scores = []
    for fold, (train_idx, valid_idx) in enumerate(gkf):
    # for fold, (train_idx, valid_idx) in enumerate(skf):
        logger.info(f'Fold No. {fold}')
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]

        # train_qa_id = df_train[['id', 'id_sub', 'label']].iloc[train_idx]

        # 通过反向翻译进行样本增强（只增强正样本）
        # 获得训练集样本的(id, id_sub)
        # train_id_set = set([f'{x[0]},{x[1]}' for x in df_train.iloc[train_idx][['id', 'id_sub']].to_numpy()])   
        # # 从增强样本中找出训练集中出现的样本
        # mask = df_train_ex[['id', 'id_sub']].apply(lambda x: f'{x["id"]},{x["id_sub"]}' in train_id_set, axis=1)    
        # df_train_fold = df_train_ex[mask]

        # train_inputs = compute_input_arrays(df_train_fold, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
        # train_outputs = compute_output_arrays(df_train_fold, output_categories)

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

        device = torch.device(f"cuda:{kwargs['device']}")
        model = BertForHouseQA().cuda(device)

        # 找到分数最高的checkpoint文件并加载
        # best_score_ = max([float(x.name[len(MODEL_NAME)+1:-3]) for x in CHECKPOINT_PATH.iterdir() if x.is_file()])
        # best_ckpt_path = CHECKPOINT_PATH/f'{MODEL_NAME}_{best_score_}.pt'
        # ckpt = torch.load(best_ckpt_path)
        # model.load_state_dict(ckpt['model_state_dict'])

        # List all modules inside the model.
        logger.info('Model modules:')
        for i, m in enumerate(model.named_children()):
            logger.info('{} -> {}'.format(i, m))

        # # Get the number of total parameters.
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel()
        #                     for p in model.parameters() if p.requires_grad)

        # logger.info("Total params: {:,}".format(total_params))
        # logger.info("Trainable params: {:,}".format(trainable_params))

        # 使用HingeLoss
        # criterion = torch.nn.MarginRankingLoss(margin=1.0)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        mode='min',
        #                                                        patience=8,
        #                                                        verbose=True
        #                                                        )
        # best_score = 0.0
        stopper = EarlyStopping(patience=kwargs['patience'], mode='max')
        for epoch in range(kwargs['epoch']):
            pass
            # =======================Training===========================
            # Set model to train mode.
            model.train()
            steps = int(np.ceil(len(train_set) / kwargs['batch_size']))
            pbar = tqdm(desc='Epoch {}, loss {}'.format(epoch, 'NAN'),
                        total=steps)
            for i, sample in enumerate(train_loader):
                x, y = sample[0].cuda(device).long(), sample[1].cuda(device).long()
                optimizer.zero_grad()

                model_outputs = model(x)    # [batch_size, 2]
                loss = criterion(model_outputs, y)

                # 使用 HingeLoss
                # loss_dict = {qid: [] for qid in train_qa_id['id']}  # 对于每个q，保存a预测为1的概率
                # pos_idx_dict = {qid: [] for qid in train_qa_id['id']}   # 对于每个q，保存正标签a所在的位置
                # for i, (qid, aid, label) in enumerate(train_qa_id.itertuple(index=False, name=None)):
                #     loss_dict[qid].append(model_outputs[i][1].item())  # qa对输出为1的概率
                #     if label == 1:
                #         pos_idx_dict[qid].append(aid)
                # losses = []
                # for qid, a_probs in loss_dict.items():
                #     a_probs = torch.tensor(a_probs)     # [a_num]
                #     pos_indices = pos_idx_dict[qid]
                #     num_neg_samples = len(a_probs) - len(pos_indices)
                #     for idx in pos_indices:
                #         pos_prob = torch.tensor(a_probs[idx])   # 一个正标签a的预测概率
                #         input1 = pos_prob.repeat(num_neg_samples)   # [num_neg_samples]
                #         input2 = torch.cat([a_probs[:idx], a_probs[idx+1:]])    # [num_neg_samples]
                #         target = torch.ones(shape=[num_neg_samples])
                #         losses.append(criterion(input1, input2, target))
                # loss = torch.mean(losses)

                loss.backward()
                optimizer.step()
                pbar.set_description(
                    'Epoch {}, train loss {:.4f}'.format(epoch, loss.item()))
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
                steps = int(np.ceil(len(valid_set) / 512))
                pbar = tqdm(desc='Validating', total=steps)
                for i, sample in enumerate(valid_loader):
                    y_true_local = sample[1].numpy()
                    x, y_true = sample[0].cuda(
                        device).long(), sample[1].cuda(device).long()

                    model_outputs = model(x)
                    loss = criterion(model_outputs, y_true).cpu().detach().item()
                    # y_pred = outputs.argmax(dim=1).cpu().numpy()
                    y_pred = F.softmax(model_outputs.cpu().detach(), dim=1).numpy()
                    valid_loss.append(loss)
                    valid_pred.append(y_pred)
                    valid_true.append(y_true_local)
                    pbar.update()
            pbar.close()
            valid_loss = np.asarray(valid_loss).mean()
            valid_pred = np.concatenate(valid_pred, axis=0)
            valid_true = np.concatenate(valid_true, axis=0)

            valid_pred_label = np.argmax(valid_pred, axis=1)
            valid_auc = roc_auc_score(valid_true, valid_pred_label)
            # valid_f1 = f1_score(valid_true, valid_pred)
            valid_p, valid_r, valid_f1, _ = precision_recall_fscore_support(valid_true, valid_pred_label, average='binary')

            # Apply ReduceLROnPlateau to the lr.
            # scheduler.step(valid_loss)

            logger.info(
            "Epoch {}, valid loss {:.5f}, valid P {:.4f}, valid R {:.4f}, valid f1 {:.4f}, valid auc {:.4f}".format(
                epoch, valid_loss, valid_p, valid_r, valid_f1, valid_auc)   
            )
            logger.info('Confusion Matrix: ')
            logger.info(confusion_matrix(y_true=valid_true, y_pred=valid_pred_label, normalize='all'))
            # if valid_auc > best_score:
                # best_score = valid_auc
            stop_flag, best_flag = stopper.step(valid_f1)
            if best_flag:
            # 保存目前的最佳模型
                torch.save(
                    {
                        "model_name": "BertForHouseQA",
                        "epoch": epoch,
                        "valid_loss": valid_loss,
                        "valid_f1": valid_f1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict()
                    },
                    f=os.path.join(CHECKPOINT_PATH, f"{MODEL_NAME}_{fold}_{epoch}_{stopper.best_score}.pt"),
                )
                logger.info("A best score! Saved to checkpoints.")
                # 保存每个验证折的预测值，用作最后整个训练集的f1评估
                all_pred[valid_idx] = valid_pred
                all_true[valid_idx] = valid_true
            if stop_flag:
                logger.info("Stop training due to early stopping.")
                # 终止训练
                break
            # 保存每个验证折的预测值，用作最后整个训练集的f1评估
            # oof[valid_idx] = valid_pred
            # valid_f1, _ = search_f1(valid_outputs, valid_pred)  # 寻找最佳分类阈值和f1 score
            # print('Valid f1 score = ', valid_f1)
            # ==========================================================

        # 每折训练结束后，进行一次预测
        # best_scores.append(stopper.best_score)
        # =======================Prediction========================
        # Set model to evaluation mode.
        # model.eval()
        # with torch.no_grad():
        #     # Prediction step
        #     test_pred = []
        #     steps = int(np.ceil(len(test_set) / 512))
        #     for i, sample in tqdm(enumerate(test_loader), desc='Predicting', total=steps):
        #         # y_true_local = sample[1].numpy()
        #         x, y_true = sample[0].cuda(
        #             device).long(), sample[1].cuda(device).float()

        #         model_outputs = model(x)
        #         # loss = criterion(model_outputs, y_true.unsqueeze(-1)).cpu().detach().item()
        #         # y_pred = outputs.argmax(dim=1).cpu().numpy()
        #         y_pred = model_outputs.cpu().detach().numpy()
        #         test_pred.append(y_pred)
        # test_pred = np.concatenate(test_pred, axis=0)
        # test_preds.append(test_pred)
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

    # 结束后，评估整个训练集
    all_pred = np.argmax(all_pred, axis=1)
    all_auc = roc_auc_score(all_true, all_pred)
    all_p, all_r, all_f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary')
    logger.info(
        "all P {:.4f}, all R {:.4f}, all f1 {:.4f}, all auc {:.4f}".format(
            all_p, all_r, all_f1, all_auc)
        )
    logger.info('Confusion Matrix: ')
    logger.info(confusion_matrix(y_true=all_true, y_pred=all_pred, normalize='all'))
    
    return all_f1


def predict_pytorch(**kwargs):
    test_inputs = kwargs['test_inputs']
    test_set = HouseDataset(test_inputs, np.zeros_like(test_inputs[0])) # 测试集没有标签
    test_loader = DataLoader(test_set,
                            batch_size=512)

    device = torch.device(f"cuda:{kwargs['device']}")
    model = BertForHouseQA().cuda(device)

    model_name = kwargs['model_name']
    time_str = kwargs['time_str']
    checkpoint_path = DATA_PATH / f"model_record/{model_name}/{time_str}" 
    ckpt_paths = [x for x in checkpoint_path.iterdir() if x.is_file() and x.suffix != '.log']

    # 找出保存的模型在每个fold训练到的最大epoch
    fold2epoch = defaultdict(int)
    for path in ckpt_paths:
        fold, epoch = str(path.name)[len(model_name)+1:].split('_')[:2]
        fold = int(fold)
        epoch = int(epoch)
        if fold2epoch[fold] < epoch:
            fold2epoch[fold] = epoch

    test_preds = []

    # 找到每个fold分数最高的checkpoint文件并加载
    for fold, epoch in fold2epoch.items():
        # pattern = f'^{model_name}_{fold}_{epoch}'
        prefix = f'{model_name}_{fold}_{epoch}'
        # best_ckpt_path = [re.match(pattern, str(x)) for x in ckpt_paths][0]
        best_ckpt_path = [x for x in ckpt_paths if str(x.name).startswith(prefix)][0]
        ckpt = torch.load(best_ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])

        # =======================Prediction========================
        # Set model to evaluation mode.
        model.eval()
        with torch.no_grad():
            # Prediction step
            test_pred = []
            steps = int(np.ceil(len(test_set) / 512))
            for i, sample in tqdm(enumerate(test_loader), desc='Predicting', total=steps):
                # y_true_local = sample[1].numpy()
                x, y_true = sample[0].cuda(
                    device).long(), sample[1].cuda(device).float()

                model_outputs = model(x)
                # loss = criterion(model_outputs, y_true.unsqueeze(-1)).cpu().detach().item()
                # y_pred = outputs.argmax(dim=1).cpu().numpy()
                y_pred = model_outputs.cpu().detach().numpy()
                test_pred.append(y_pred)
        test_pred = np.concatenate(test_pred, axis=0)
        test_preds.append(test_pred)
    
    return test_preds



all_f1 = train_pytorch(batch_size=128, epoch=15, lr=2e-5, weight_decay=1e-3, n_splits=10, patience=8, device=0, inputs=inputs, 
                                        outputs=outputs, test_inputs=test_inputs)
# test_time_str = '2020-11-06-11:25:02'
# test_preds = predict_pytorch(test_inputs=test_inputs, device=0, model_name=MODEL_NAME, time_str=test_time_str) 

                                

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
# best_score, best_t = search_f1(outputs, oof)

# best_score = np.average(best_scores)
# logger.info(f'Best score: {best_score}')
# %%
sub = np.average(test_preds, axis=0) 
sub = np.argmax(sub, axis=1)
# sub = sub > best_t  # 用该分类阈值来输出测试集
df_test['label'] = sub.astype(int)
# df_test[['id','id_sub','label']].to_csv(f'{CHECKPOINT_PATH}/submission_beike_{all_f1}.csv',index=False, header=None,sep='\t')
fpath = DATA_PATH / f"model_record/{MODEL_NAME}/{test_time_str}/submission_beike_0.7939.csv"
df_test[['id','id_sub','label']].to_csv(fpath,index=False, header=None,sep='\t')

# %%



