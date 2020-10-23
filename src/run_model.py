# coding=utf-8
import os
import logging
import argparse
from model.qa_mode.qa_model import QaModel
from utils.set_random_seed import setup_seed


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
''' 定义参数和默认值 '''
parser = argparse.ArgumentParser()

parser.add_argument('--machine', default='85', choices=['85', '86', '87'])
parser.add_argument('--pretrain_bert_model', default='google', choices=['google'])
parser.add_argument('--data_type', default='qa_data3', choices=['qa_data', 'qa_data2', 'qa_data3'])
parser.add_argument('--model_name', default='bert_cls_model', choices=['bert_cls_model'])

parser.add_argument('--exp_name', default='qa1')
parser.add_argument('--run_mode', default='train', choices=['train', 'get_result', 'train_k_fold', 'get_result'])


parser.add_argument('--freeze', type=bool, default=False)
parser.add_argument('--freeze_layer', type=int, default=-1)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--seed', type=int, default=1234)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--accumulation_loss_step', type=int, default=1)
parser.add_argument('--batch_num_per_epoch', type=int, default=-1)
parser.add_argument('--epoch_num', type=int, default=20)
parser.add_argument('--min_bz_per_gpu', type=int, default=1)

parser.add_argument('--class_num', type=int, default=1)

parser.add_argument('--load_pretrain_model', type=bool, default=False)

parser.add_argument('--max_seq_len', type=int, default=100)
parser.add_argument('--max_q_len', type=int, default=-1)
parser.add_argument('--max_d_len', type=int, default=-1)
parser.add_argument('--max_para_num', type=int, default=-1)
parser.add_argument('--overlap', type=bool, default=False)

parser.add_argument('--eval_train', type=bool, default=True)    # 评估训练集
parser.add_argument('--eval', type=bool, default=True)  # 评估测试集
parser.add_argument('--have_val', type=bool, default=False)     # 输出submission时设为True
parser.add_argument('--always_save', type=bool, default=False)

parser.add_argument('--dropout', type=float, default=0.5)


parser.add_argument('--eval_mode', default='max', choices=['max', 'mean'])
parser.add_argument('--parallel', type=bool, default=True)
parser.add_argument('--bert_hidden_size', type=int, default=768)


parser.add_argument('--optimizer', default='adam')
parser.add_argument('--shuffle', type=bool, default=False)


args = parser.parse_args()

''' 设置种子 '''
if args.seed != -1:
    setup_seed(args.seed)

''' 设置基本打印信息 '''
logging.basicConfig(level=logging.INFO)

''' 设置工程根目录、数据根目录 '''
abs_file_path = os.path.dirname(__file__)
ROOT_DIR = os.path.join(abs_file_path, '../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')



''' 定义模型保存目录 '''
MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'model_record')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH,
                               args.data_type,
                               args.model_name,
                               ((args.exp_name + '_exp%d' % (args.exp_id))
                               if args.model_name=='bert_pli_model'  else args.exp_name))

''' BERT预训练参数根目录 '''
BERT_PRETRAIN_MODEL_DIR = os.path.join(DATA_DIR, 'pretrain_model')

''' 定义BERT预训练数据目录 '''
if args.pretrain_bert_model == 'google':
    # BERT_PRETRAIN_MODEL = os.path.join(BERT_PRETRAIN_MODEL_DIR, 'bert-base-chinese')
    BERT_PRETRAIN_MODEL = 'bert-base-chinese'
else:
    BERT_PRETRAIN_MODEL = None
    RuntimeError('None pretrain model specific !!')



if __name__ == '__main__':

    bert_for_ir = QaModel(bert_model_dir= BERT_PRETRAIN_MODEL,
                          save_model_path=MODEL_SAVE_PATH,
                          eval_mode=args.eval_mode,
                          args=args)

    if args.run_mode == 'train':
        bert_for_ir.train_loop(num_epoch=args.epoch_num)
    elif args.run_mode == 'show_weight':
        bert_for_ir.show_weight()
    elif args.run_mode == 'train_k_fold':
        bert_for_ir.train_k_fold(args.epoch_num)
    elif args.run_mode == 'get_result':
        bert_for_ir.get_best_result()
    else:
        RuntimeError('No run mode found !!')


