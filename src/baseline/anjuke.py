# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from collections import defaultdict
import logging
import re
import time
import pandas as pd
import numpy as np
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
from transformers.file_utils import ModelOutput
from house_dataset import HouseDataset, BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig, AutoModel, AutoTokenizer
import random
from early_stopping import EarlyStopping


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

# df_train = pd.read_csv(DATA_PATH/'anjuke/anjuke_full.tsv', sep='\t', header=None, 
                                # names=['id', 'q1', 'id_sub', 'q2', 'label', 'city'])
df_train = pd.read_csv(DATA_PATH/'anjuke/anjuke_pos.tsv', sep='\t', header=None, 
                                names=['id', 'q1', 'id_sub', 'q2', 'label', 'city'])



STOPWORS_PATH = PROJECT_ROOT_PATH / 'src/utils/stopwords/HIT_stop_words.txt'
stopwords = set()
with open(STOPWORS_PATH, 'r') as f:
    stopwords.update([x.strip() for x in f.readlines()])


# %%
# PATH = './'
# BERT_PATH = './data/pretrain_model'
# WEIGHT_PATH = './'
MAX_SEQUENCE_LENGTH = 100
# 'chinese_roberta_wwm_ext_pytorch', 'chinese_roberta_wwm_large_ext_pytorch', 'bert-base-chinese', 'chinese_wwm_ext_pytorch', 'ernie'
MODEL_NAME = 'chinese_wwm_ext_pytorch'
TIME_STR = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
CHECKPOINT_PATH = DATA_PATH / f"anjuke/model_record/{MODEL_NAME}/{TIME_STR}"
PRETRAIN_MODEL_PATH = PROJECT_ROOT_PATH/'./data/pretrain_model'
input_categories = ['q1','q2']
output_categories = 'label'

print('train shape =', df_train.shape)

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

        # inputs = tokenizer.encode_plus(str1, str2,
        #     add_special_tokens=True,
        #     max_length=length,
        #     # truncation_strategy='longest_first',
        #     truncation='longest_first',
        #     padding='max_length'
        #     )
        # Tokenizer的__call__方法可以直接处理str batch或者单独的str
        inputs = tokenizer(str1, str2,
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


def get_overlap_count(a, b):
    unique_a, counts_a = np.unique(a, return_counts=True)
    unique_b, counts_b = np.unique(b, return_counts=True)
    
    unique_id, counts_id = np.unique(np.concatenate([unique_a, unique_b]), return_counts=True)
    overlap_id = unique_id[counts_id > 1]   # 获得a和b中共现的字符/词/id
    if overlap_id.shape[0] == 0:
        return 0     # 返回0
    
    overlap_counts = []
    for i in overlap_id:
        idx_a = np.argmax((unique_a == i).astype(np.int32))
        idx_b = np.argmax((unique_b == i).astype(np.int32))
        overlap_counts.append(counts_a[idx_a])
        overlap_counts.append(counts_b[idx_b])
    overlap_count = np.stack(overlap_counts).sum()

    # overlap_id = set(unique_a).intersection(set(unique_b))
    # if len(overlap_id) == 0:
        # return 0
    # 
    # unique2count_a = dict(zip(unique_a, counts_a))
    # unique2count_b = dict(zip(unique_b, counts_b))
# 
    # overlap_count = 0
    # for i in overlap_id:
        # overlap_count += (unique2count_a[i] + unique2count_b[i])
    return overlap_count


def get_overlap_feature(str_a, str_b):
    import jieba
    char_a = list(str_a)
    char_b = list(str_b)

    word_a = [x for x in jieba.cut(str_a) if x not in stopwords]
    word_b = [x for x in jieba.cut(str_b) if x not in stopwords]

    overlap_count_char = get_overlap_count(char_a, char_b)
    overlap_count_word = get_overlap_count(word_a, word_b)

    return np.asarray([overlap_count_char, overlap_count_word])


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    input_overlap = []
    # 使用tokenizer的批处理优化速度
    dataset = BaseDataset(df['q1'], df['q2'])
    batch_size = 2048
    loader = DataLoader(dataset, batch_size=batch_size)
    steps = int(np.ceil(len(dataset) / batch_size))
    pbar = tqdm(desc='Computing input arrays', total=steps)
    for i, sample in enumerate(loader):
        q_batch, a_batch = sample[0], sample[1]
        ids_q, masks_q, segments_q = _convert_to_transformer_inputs(q_batch, a_batch, tokenizer, max_sequence_length)
        # TODO: 添加overlap特征
        overlap_feat = np.asarray([get_overlap_feature(q, a) for q, a in zip(q_batch, a_batch)])
        
        input_ids_q.extend(ids_q)
        input_masks_q.extend(masks_q)
        input_segments_q.extend(segments_q)
        input_overlap.extend(overlap_feat)
        pbar.update()
    pbar.close()

    # for _, instance in tqdm(df[columns].iterrows()):
    #     q, a = instance.q1, instance.q2

    #     ids_q, masks_q, segments_q = _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)
        
    #     input_ids_q.append(ids_q)
    #     input_masks_q.append(masks_q)
    #     input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32),
            ], np.asarray(input_overlap, dtype=np.int32)


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(30,70):
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

if MODEL_NAME == 'ernie':
    tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-1.0', cache_dir=PRETRAIN_MODEL_PATH/MODEL_NAME)
else:
    tokenizer = BertTokenizer.from_pretrained(str(PRETRAIN_MODEL_PATH/MODEL_NAME))


inputs_path = DATA_PATH / 'anjuke/inputs_pos.npy'
outputs_path = DATA_PATH / 'anjuke/outputs_pos.npy'

# 保存处理好的数据集
if not outputs_path.exists():
    outputs = compute_output_arrays(df_train, output_categories)
    np.save(outputs_path, outputs)
else:
    outputs = np.load(outputs_path, allow_pickle=True)

if not inputs_path.exists():
    inputs, inputs_overlap = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    np.save(inputs_path, inputs)
    np.save(inputs_path.parent / 'inputs_overlap_pos.npy', inputs_overlap)
else:
    inputs = np.load(inputs_path, allow_pickle=True)
    inputs_overlap = np.load(inputs_path.parent / 'inputs_overlap.npy', allow_pickle=True)

# inputs_path = Path(RAW_DATA_PATH/'./train/inputs.npy')
# outputs_path = Path(RAW_DATA_PATH/'./train/outputs.npy')
# test_inputs_path = Path(RAW_DATA_PATH/'./test/test_inputs.npy')

# # 保存处理好的数据集
# if not outputs_path.exists():
#     outputs = compute_output_arrays(df_train, output_categories)
#     np.save(outputs_path, outputs)
# else:
#     outputs = np.load(outputs_path, allow_pickle=True)

# if not inputs_path.exists():
#     inputs, inputs_overlap = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
#     np.save(inputs_path, inputs)
#     np.save(inputs_path.parent / 'inputs_overlap.npy', inputs_overlap)
# else:
#     inputs = np.load(inputs_path, allow_pickle=True)
#     inputs_overlap = np.load(inputs_path.parent / 'inputs_overlap.npy', allow_pickle=True)


# Pytorch版BERT模型
class BertForHouseQA(nn.Module):
    def __init__(self):
        super(BertForHouseQA, self).__init__()
        # self.bert = BertModel.from_pretrained(os.path.join(PRETRAIN_MODEL_PATH, 'bert-base-chinese'), 
        if MODEL_NAME == 'ernie':
            self.bert = AutoModel.from_pretrained('nghuyong/ernie-1.0',
                    cache_dir = PRETRAIN_MODEL_PATH / MODEL_NAME,
                    output_hidden_states=False, 
                    output_attentions=False)
        else:
            self.bert = BertModel.from_pretrained(os.path.join(PRETRAIN_MODEL_PATH, MODEL_NAME), 
                        output_hidden_states=False, 
                        output_attentions=False)
        
                    
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)
        # self.fc = nn.Linear(3*self.bert.config.hidden_size, 2)
        # self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        # self.fc = nn.Linear(self.bert.config.hidden_size + 1, 2)
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.W = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor):
        hidden_state = self.bert(
            input_ids=x[:, 0, :], attention_mask=x[:, 1, :], token_type_ids=x[:, 2, :]
        )[0]    # [b, seq, hid]

        feat_cls = hidden_state[:, 0, :]    # [CLS]
        # feat_mean = torch.mean(hidden_state, dim=1) # [b, hid]
        # feat_max, _ = torch.max(hidden_state, dim=1) # [b, hid]
        # feat = torch.cat([feat_cls, feat_mean, feat_max], dim=-1) # [b, 3*hid]

        # =================================================
        # feat_a = torch.mean(hidden_state[mask_a][1:-1], dim=1).unsqueeze(1)     # [b, 1, hid]
        # feat_b = torch.mean(hidden_state[mask_b][:-1], dim=1).unsqueeze(-1)    # [b, hid, 1]
        # sim_ab = torch.squeeze(self.W(feat_a) * feat_b) # [b, 1, 1]
        # =================================================

        feat = self.dropout(feat_cls)
        # logit = self.fc(torch.cat([self.relu(feat), overlap_count], dim=1))
        logit = self.fc(self.relu(feat))
        # logit = self.fc(self.tanh(feat))
        # logit = self.fc(feat)
        # logit = self.sigmoid(self.fc(feat))

        return logit


class BertClsToReg(nn.Module):
    def __init__(self, org_bert: BertForHouseQA):
        super(BertClsToReg, self).__init__()
        # self.bert = BertModel.from_pretrained(os.path.join(PRETRAIN_MODEL_PATH, 'bert-base-chinese'), 
        self.org_bert = org_bert
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.org_bert.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):
        name, bert = next(self.org_bert.named_children())
        assert name == 'bert'
        hidden_state = bert(
            input_ids=x[:, 0, :], attention_mask=x[:, 1, :], token_type_ids=x[:, 2, :]
        )[0]    # [b, seq, hid]
        
        feat_cls = hidden_state[:, 0, :]    # [CLS]
        
        feat = self.dropout(feat_cls)
        logit = self.sigmoid(self.fc(feat))

        return logit


def get_hinge_loss(model_outputs, qa_id, criterion):
    qids = set(qa_id[:, 0])
    losses = []
    for qid in qids:
        # 属于当前qid的mask
        mask = qa_id[:, 0] == qid
        # 属于当前qid并且标签为正或负的mask
        pos_mask = mask & (qa_id[:, 2] == 1)
        neg_mask = mask & (qa_id[:, 2] == 0)
        if (pos_mask).sum().item() == 0:
            continue    # 该问题没有正确答案，没法计算hingeloss
        pos_probs = model_outputs[pos_mask]
        if (neg_mask).sum().item() == 0:
            # 该问题没有错误答案
            # 在当前答案集合中随机采样负样本(排除当前问题的正样本)
            candidate_probs = model_outputs[~pos_mask]
            num_candidate = candidate_probs.shape[0]
            # idx = torch.multinomial(torch.ones([num_candidate]), num_samples=4)
            idx = torch.randint(high=num_candidate, size=(4,))
            neg_probs = candidate_probs[idx]
        else:
            neg_probs = model_outputs[neg_mask]
        
        input1 = pos_probs.repeat(neg_probs.shape[0], 1)
        input2 = neg_probs.repeat(pos_probs.shape[0], 1)
        target = torch.ones_like(input1)
        loss = criterion(input1, input2, target)
        # if torch.isnan(loss).item() == True:
        #     continue
        losses.append(loss)
    losses = torch.stack(losses)
    loss = torch.sum(losses)
    return loss


def train_pytorch(**kwargs):
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

    # 调用logging.basicConfig会给进程添加一个root logger，这样其他模块中logger的日志才会显示到console当中
    # （子logger传到root logger，root logger通过他自带的StreamHandler输出）。
    # 如果不调用logging.basicConfig，必须得每个子logger配置一个StreamHandler，很麻烦
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    formater = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Print logs to the terminal.
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formater)
    # # Save logs to file.
    log_path = CHECKPOINT_PATH / 'train.log'
    file_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formater)
    
    # logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    inputs = kwargs['inputs']
    outputs = kwargs['outputs']
    # test_inputs = kwargs['test_inputs']
    
    # gkf = GroupKFold(n_splits=kwargs['n_splits']).split(X=df_train.q2, groups=df_train.id)

    sss = StratifiedShuffleSplit(n_splits=kwargs['n_splits'], test_size=0.2).split(X=df_train.q2, 
            y=df_train.label)
    # skf = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=RANDOM_SEED).split(X=df_train.q2, y=outputs)

    # oof = np.zeros((len(df_train),1))
    all_pred = np.zeros(shape=(len(df_train), 2))     # 分类任务
    # all_pred = np.zeros(shape=(len(df_train)))  # 回归任务
    all_true = np.zeros(shape=(len(df_train)))
    for fold, (train_idx, valid_idx) in enumerate(sss):
    # for fold, (train_idx, valid_idx) in enumerate(skf):
        logger.info(f'Fold No. {fold}')
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]

        train_qa_id = df_train[['id', 'id_sub', 'label']].iloc[train_idx]

        # 通过反向翻译进行样本增强（只增强正样本）
        # 获得训练集样本的(id, id_sub)
        # train_id_set = set([f'{x[0]},{x[1]}' for x in df_train.iloc[train_idx][['id', 'id_sub']].to_numpy()])   
        # # 从增强样本中找出训练集中出现的样本
        # mask = df_train_ex[['id', 'id_sub']].apply(lambda x: f'{x["id"]},{x["id_sub"]}' in train_id_set, axis=1)    
        # df_train_fold = df_train_ex[mask]

        # 获得训练集样本的(id, id_sub)
        # train_id_set = set([f'{x[0]},{x[1]}' for x in df_train.iloc[train_idx][['id', 'id_sub']].to_numpy()])   
        # # 从增强样本中找出训练集中出现的样本
        # mask = df_train_aug[['id', 'id_sub']].apply(lambda x: f'{x["id"]},{x["id_sub"]}' in train_id_set, axis=1)    
        # df_train_fold = df_train_aug[mask]

        # train_inputs = compute_input_arrays(df_train_fold, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
        # train_outputs = compute_output_arrays(df_train_fold, output_categories)

        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]
        valid_qa_id = df_train[['id', 'id_sub', 'label']].iloc[valid_idx]

        train_set = HouseDataset(train_inputs, train_outputs, train_qa_id)
        valid_set = HouseDataset(valid_inputs, valid_outputs, valid_qa_id)
        # test_set = HouseDataset(test_inputs, np.zeros_like(test_inputs[0])) # 测试集没有标签

        logger.info('Train set size: {}, valid set size {}'.format(
            len(train_set), len(valid_set)))

        train_loader = DataLoader(train_set,
                                batch_size=kwargs['batch_size'],
                                shuffle=True  # 如果使用分类训练，建议True
                                )

        valid_loader = DataLoader(valid_set,
                                batch_size=kwargs['valid_batch_size'])

        # test_loader = DataLoader(test_set,
                                # batch_size=512)

        device = torch.device(f"cuda:{kwargs['device']}")
        model = BertForHouseQA().cuda(device)

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
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'])
        logger.info('Optimizer:')
        logger.info(optimizer)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        mode='min',
        #                                                        patience=8,
        #                                                        verbose=True
        #                                                        )
        # best_score = 0.0
        stopper = EarlyStopping(patience=kwargs['patience'], mode='max')
        ckpt_path = None
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
                # CrossEntropy
                loss = criterion(model_outputs, y)
                # MSE
                # loss = criterion(model_outputs, y.float().unsqueeze(-1))

                # 使用 HingeLoss
                # train_qa_id_sub = sample[2].numpy()
                # loss = get_hinge_loss(model_outputs, train_qa_id_sub, criterion)

                # 使用SCL
                # inners = torch.do
                
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
                steps = int(np.ceil(len(valid_set) / kwargs['valid_batch_size']))
                pbar = tqdm(desc='Validating', total=steps)
                for i, sample in enumerate(valid_loader):
                    y_true_local = sample[1].numpy()
                    x, y_true = sample[0].cuda(
                        device).long(), sample[1].cuda(device).long()

                    model_outputs = model(x)
                    # MSELoss
                    # loss = criterion(model_outputs, y_true.float().unsqueeze(-1)).cpu().detach().item()
                    # HingeLoss
                    # valid_qa_id_sub = sample[2].numpy()
                    # loss = get_hinge_loss(model_outputs, valid_qa_id_sub, criterion)
                    # y_pred = model_outputs.cpu().detach().squeeze(-1).numpy()
                    # CrossEntropy
                    loss = criterion(model_outputs, y_true).cpu().detach().item()
                    y_pred = F.softmax(model_outputs.cpu().detach(), dim=1).numpy()
                    
                    valid_loss.append(loss)
                    valid_pred.append(y_pred)
                    valid_true.append(y_true_local)
                    pbar.update()
            pbar.close()
            valid_loss = np.asarray(valid_loss).mean()
            valid_pred = np.concatenate(valid_pred, axis=0)
            valid_true = np.concatenate(valid_true, axis=0)

            # 如果使用回归模型
            # valid_f1, thr = search_f1(valid_true, valid_pred)
            # logger.info("Epoch {}, valid loss {:.5f}, valid f1 {:.4f}".format(epoch, valid_loss, valid_f1)))

            # 如果使用分类模型
            valid_pred_label = np.argmax(valid_pred, axis=1)
            valid_auc = roc_auc_score(valid_true, valid_pred_label)
            valid_p, valid_r, valid_f1, _ = precision_recall_fscore_support(valid_true, valid_pred_label, average='binary')

            # Apply ReduceLROnPlateau to the lr.
            # scheduler.step(valid_loss)

            logger.info(
            "Epoch {}, valid loss {:.5f}, valid P {:.4f}, valid R {:.4f}, valid f1 {:.4f}, valid auc {:.4f}".format(
                epoch, valid_loss, valid_p, valid_r, valid_f1, valid_auc)   
            )
            logger.info('Confusion Matrix: ')
            logger.info(confusion_matrix(y_true=valid_true, y_pred=valid_pred_label, normalize='all'))
            stop_flag, best_flag = stopper.step(valid_f1)
            if best_flag:
                # 删除之前保存的模型
                if ckpt_path is not None:
                    ckpt_path.unlink()
                ckpt_path = CHECKPOINT_PATH / f"{MODEL_NAME}_{fold}_{epoch}_{stopper.best_score}.pt"
            # 保存目前的最佳模型
                torch.save(
                    {
                        "model_name": "BertForHouseQA",
                        "epoch": epoch,
                        "valid_loss": valid_loss,
                        "valid_f1": valid_f1,
                        "model_state_dict": model.state_dict(),
                        # "optimizer_state_dict": optimizer.state_dict(),
                        # "thr": thr
                        # 'scheduler_state_dict': scheduler.state_dict()
                    },
                    f=ckpt_path,
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

    # 结束后，评估整个训练集
    # CrossEntropy
    all_pred = np.argmax(all_pred, axis=1)
    all_auc = roc_auc_score(all_true, all_pred)
    all_p, all_r, all_f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary')
    logger.info(
        "all P {:.4f}, all R {:.4f}, all f1 {:.4f}, all auc {:.4f}".format(
            all_p, all_r, all_f1, all_auc)
        )
    logger.info('Confusion Matrix: ')
    logger.info(confusion_matrix(y_true=all_true, y_pred=all_pred, normalize='all'))
    # MSELoss
    # all_f1, all_thr = search_f1(all_true, all_pred)
    # logger.info("All f1 {:.4f}, all thr {:.4f}".format(all_f1, all_thr))
    return all_f1, CHECKPOINT_PATH


if __name__ == "__main__":
    # ERNIE需要的初始学习率较高，参考https://github.com/ymcui/Chinese-BERT-wwm
    # 由于BERT/BERT-wwm使用了维基百科数据进行训练，故它们对正式文本建模较好；
    # 而ERNIE使用了额外的百度贴吧、知道等网络数据，它对非正式文本（例如微博等）建模有优势。
    # 在长文本建模任务上，例如阅读理解、文档分类，BERT和BERT-wwm的效果较好。
    
    # 使用ERNIE请删除.npy文件并重新生成，因为ERNIR的vocab和wwm系列模型不一样
    all_f1, checkpoint_path = train_pytorch(batch_size=128, valid_batch_size=512, epoch=6, lr=2e-5, weight_decay=1e-3, 
                                        n_splits=1, patience=3, device=1, inputs=inputs, 
                                        outputs=outputs)

