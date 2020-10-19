# coding=utf-8

import sys
sys.path.append('../../')
import os
import os.path as oph
import torch
import logging
from dataset.qa_data.dataset import QaDataset
from model.metrics.eval_callback import EvaluateMetrics, Metrics, SaveModelCallback
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.nn as nn
from .model.bert_cls_model import BertCLSModel
import torch.optim as optim
import time


class QaModel():
    def __init__(self, bert_model_dir,
                 save_model_path,
                 eval_mode ,
                 args):

        ''' 创建log文件夹 '''
        self.log_file = oph.join(save_model_path, 'log.txt')

        ''' 创建summary对象 '''
        summary_dir = oph.join(save_model_path, 'summary')
        self.summary_writer = SummaryWriter(summary_dir)

        ''' 创建checkpoint文件夹 '''
        checkpoint = os.path.join(save_model_path, 'checkpoint')
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.model_save_path = checkpoint

        ''' 创建logger对象 '''
        FORMAT = '%(filename)s->%(funcName)s(%(levelname)s): %(message)s'
        format = logging.Formatter(FORMAT)
        logger = logging.getLogger('Bert4Ir')
        logger.setLevel(logging.DEBUG)
        fhl = logging.FileHandler(filename=self.log_file, encoding='utf-8')
        fhl.setLevel(logging.DEBUG)
        fhl.setFormatter(format)
        logger.addHandler(fhl)
        self.log = logger

        localtime = time.asctime(time.localtime(time.time()))

        self.log.info('\n')
        self.log.info('Exp: %s ---------------------------------%s----------------------------------------' % (args.exp_name, localtime))
        ''' 保存超参 '''
        self.args = args
        logger.info(args)

        ''' 查看GPU数量 '''
        logger.info('GPU num: %d' % (torch.cuda.device_count()))

        ''' 根据模型和数据类型加载数据集 '''
        if args.data_type == 'qa_data':
            self.dataset = QaDataset(vocab_dir=bert_model_dir, args=args)
        else:
            RuntimeError('None model found !!')
        self.train_generator = self.dataset.train_generator
        self.test_generator = self.dataset.test_generator
        if args.have_val == True:
            self.dev_generator = self.dataset.dev_generator

        ''' 构建模型对象 '''
        if args.model_name == 'bert_cls_model':
            self.qa_model = BertCLSModel(bert_model_dir=bert_model_dir, args=args)
        else:
            RuntimeError('None model found !!')

        ''' 得到loss函数 '''
        self.loss_fun = self.qa_model.get_loss_function()

        ''' 决定是否并行运行 '''
        if args.parallel:
            self.qa_model = nn.DataParallel(self.qa_model)



        ''' 冻结最后一层encoder之前的层 '''
        if args.freeze:
            self.log.info('Freeze layer %d' % args.freeze_layer)
            self.para_freezer(args.freeze_layer)
        else:
            self.log.info('No layer was freezed.' )


        ''' 构建优化器 '''
        params = filter(lambda p:p.requires_grad, self.qa_model.parameters())
        self.optimizer = optim.Adam(params, lr=args.lr)


        ''' 构建保存模型的回调 '''
        self.save_callback = SaveModelCallback(model=self.qa_model,
                                               optimizer=self.optimizer,
                                               save_model_path=checkpoint,
                                               log=self.log, always_save=args.always_save)

        ''' 构建三个数据集的评估回调函数 '''
        if self.args.have_val == True:
            self.eval_fun = EvaluateMetrics(metric=Metrics,
                                            args=args,
                                            generator=self.dev_generator,
                                            save_model_callback=None,
                                            mode=eval_mode,
                                            log=self.log)
            self.test_fun = EvaluateMetrics(metric=Metrics,
                                            args=args,
                                            generator=self.test_generator,
                                            save_model_callback=self.save_callback,
                                            mode=eval_mode,
                                            log=self.log)
        else:
            self.test_fun = EvaluateMetrics(metric=Metrics,
                                            args=args,
                                            generator=self.test_generator,
                                            save_model_callback=self.save_callback,
                                            mode=eval_mode,
                                            log=self.log)



        if args.eval_train:
            self.train_eval_fun = EvaluateMetrics(metric=Metrics,
                                                  args=args,
                                                  generator=self.train_generator,
                                                  save_model_callback=None,
                                                  mode=eval_mode,
                                                  log=self.log)


        ''' 加载已经存在的模型 '''
        self.epoch = 0
        if os.path.exists(os.path.join(checkpoint, '0', 'ckpt.pth')):
            self.log.debug('ckpt.pth exist! Loading...')
            max_epoch = 0
            model_path = ''
            '''加载最新的，即epoch标号最大的模型'''
            for root, dirs, files in os.walk(checkpoint):
                for d in dirs:
                    epoch_num = int(d)
                    if epoch_num >= max_epoch:
                        model_path = os.path.join(root, d, 'ckpt.pth')
                        if os.path.exists(model_path):
                            max_epoch = epoch_num
                            model_path = os.path.join(root, d, 'ckpt.pth')
            self.log.debug('Load %s ' % model_path)
            checkpoint = torch.load(model_path)
            self.qa_model.load_state_dict(checkpoint['model_state_dict'])

            ''' 如果加载预训练模型参数，则不需要加载logs epoch optimizer数据 '''
            if not self.args.load_pretrain_model:

                self.save_callback.now_best_val_log=checkpoint['val_logs']
                self.epoch = checkpoint['epoch'] + 1
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to('cuda')


        self.qa_model.to('cuda')
        self.loss_fun.to('cuda')


    ''' 训练循环 '''
    def train_loop(self, num_epoch=100):
        epoch = self.epoch
        self.qa_model.zero_grad()
        self.optimizer.zero_grad()
        step = epoch*self.args.batch_num_per_epoch
        for _ in range(num_epoch - epoch):
            self.qa_model.train()
            self.log.info('\nEpoch: %d Training:----------------------------------------------------------------------' % epoch)
            batches_bar = tqdm(range(len(self.train_generator)), desc='Training')
            # batches_bar = tqdm(range(4), desc='Training')
            predictions = []
            losses = []
            labels = []
            for i in batches_bar:
                ''' 得到批次数据index '''
                if self.args.always_save:
                    index = epoch*self.args.batch_num_per_epoch + i
                else:
                    index = i

                ''' 得到批次数据 '''
                try:
                    batch_data, rep_num = self.train_generator[index]
                except StopIteration:
                    break

                batch_data_cuda = []
                ''' 将数据导入 GPU'''
                for data in batch_data:
                    batch_data_cuda.append(data.to('cuda'))
                batch_data_cuda = tuple(batch_data_cuda)

                ''' 进行一次反向传播 '''
                pre = self.qa_model(batch_data_cuda)
                bz = list(pre[0].size())[0]
                loss = self.loss_fun(pre[0][:int(rep_num*bz)], batch_data_cuda[-1][:int(rep_num*bz)])
                loss = loss / self.args.accumulation_loss_step
                loss.backward()
                if (index + 1) % self.args.accumulation_loss_step ==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                ''' 记录loss '''
                tmp_loss = loss*self.args.accumulation_loss_step
                losses.append(tmp_loss.item())
                batches_bar.set_description("Training Loss: %f" % tmp_loss.item())
                self.summary_writer.add_scalar('Train/loss', tmp_loss.item(), step)
                self.summary_writer.flush()
                step += 1

                ''' 记录每个batch的结果 '''
                bz = list(pre[1].size())[0]
                out = pre[1][:int(bz*rep_num)]
                out = out.cpu().detach().numpy()

                predictions.append(out)

                ''' 记录每个batch的label '''
                bz = list(pre[1].size())[0]
                l = batch_data_cuda[-1][:int(bz*rep_num)]
                l = l.cpu().detach().numpy()
                labels.append(l)


            '''得到该次epoch平均loss'''
            avg_loss = sum(losses)/len(losses)
            self.log.info('Average loss : %f ' % avg_loss)
            self.summary_writer.add_scalar('Train/avg_loss', avg_loss, epoch)
            self.summary_writer.flush()


            ''' 对训练数据进行一次评估 '''
            if self.args.eval_train:
                pred = np.concatenate(predictions, axis=0)
                if pred.shape[-1] >= 2:
                    pred = np.argmax(pred, axis=-1)

                self.train_eval_fun.set_thre(0.5)
                self.train_eval_fun.eval(epoch=epoch, pred=pred, step=epoch, summary_writer=self.summary_writer, mode='Train')


            if self.args.eval:
                if self.args.have_val == True:
                    ''' 对验证数据进行一次评估 '''
                    self.log.info('Epoch: %d Evaluating----------------------------------------------------------------------' % epoch)
                    self.evaluate(epoch)

                ''' 对测试数据进行评估 '''
                self.log.info('Epoch: %d Testing-------------------------------------------------------------------------' % epoch)
                self.test(epoch)

            epoch += 1
            ''' 训练完毕回调 '''
            self.train_generator.on_epoch_end_callback()

    def evaluate(self, epoch):
        self.qa_model.eval()
        with torch.no_grad():
            res = []
            for i in tqdm(range(len(self.dev_generator)), desc='Evaluating'):
                ''' 得到数据 '''
                try:
                    batch_data, rep_num = self.dev_generator[i]
                except StopIteration:
                    break

                batch_data_cuda = []
                ''' 将数据导入 GPU'''
                for data in batch_data:
                    batch_data_cuda.append(data.cuda())
                batch_data_cuda = tuple(batch_data_cuda)
                ''' 预测结果 '''
                out = self.qa_model(batch_data_cuda)[1]
                bz = list(out.size())[0]
                out = out[:int(rep_num*bz)]
                out = out.cpu().detach().numpy()
                res.append(out)
            ''' 进行评估 '''
            pred = np.concatenate(res, axis=0)
            if pred.shape[-1] >= 2:
                pred = np.argmax(pred, axis=-1)

            self.eval_fun.set_thre(0.5)
            self.eval_fun.eval(epoch=epoch, pred=pred, step=epoch, summary_writer=self.summary_writer, mode='Eval')


    def test(self, epoch):
        self.qa_model.eval()
        with torch.no_grad():
            res = []
            for i in tqdm(range(len(self.test_generator)), desc='Testing'):
                ''' 得到数据 '''
                try:
                    batch_data, rep_num = self.test_generator[i]
                except StopIteration:
                    break

                batch_data_cuda = []
                ''' 将数据导入 GPU'''
                for data in batch_data:
                    batch_data_cuda.append(data.cuda())
                batch_data_cuda = tuple(batch_data_cuda)
                ''' 预测结果 '''
                out = self.qa_model(batch_data_cuda)[1]
                bz = list(out.size())[0]
                out = out[:int(bz*rep_num)]
                out = out.cpu().detach().numpy()
                res.append(out)
            ''' 进行评估 '''
            pred = np.concatenate(res, axis=0)
            if pred.shape[-1] >= 2:
                pred = np.argmax(pred, axis=-1)

            self.test_fun.set_thre(0.5)
            self.test_fun.eval(epoch=epoch, pred=pred, step=epoch, summary_writer=self.summary_writer, mode='Test')

    def para_freezer(self, freeze_layer):
        for i, p in enumerate(self.qa_model.parameters()):
            if i <= freeze_layer:  # 196:
                p.requires_grad = False

    def show_weight(self):
        print('----------------------------------------------------')
        dict_name = list(self.qa_model.state_dict())
        for i,p in enumerate(dict_name):
            print(i, p)
        print('----------------------------------------------------')









