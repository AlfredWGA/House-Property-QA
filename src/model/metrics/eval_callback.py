# coding=utf-8

import typing
from matchzoo.engine.base_metric import BaseMetric
import numpy as np
import pandas as pd
import matchzoo as mz
import os
import torch

from .f1 import F1
from .percision import Precision
from .recall import Recall

Metrics = [
   Precision(),
   Recall(),
   F1()
]


class SaveModelCallback():
    def __init__(
        self,
        model,
        optimizer,
        save_model_path = None,
        log = None,
        always_save = False
    ):
        self.always_save = always_save
        self.log = log
        self.model = model
        self.optimizer = optimizer
        self.save_model_path = save_model_path
        self.now_best_val_log = {m: 0.0 for m in Metrics}
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

    def is_better(self, latest_val_log):
        precision = Metrics[0]
        recall = Metrics[1]
        f1 = Metrics[2]

        if latest_val_log[f1] > self.now_best_val_log[f1]:

            if latest_val_log[f1] > self.now_best_val_log[f1]:
                self.now_best_val_log[f1] = latest_val_log[f1]

            if latest_val_log[precision] > self.now_best_val_log[precision]:
                self.now_best_val_log[precision] = latest_val_log[precision]

            if latest_val_log[recall] > self.now_best_val_log[recall]:
                self.now_best_val_log[recall] = latest_val_log[recall]

            return True
        else:


            if latest_val_log[f1] > self.now_best_val_log[f1]:
                self.now_best_val_log[f1] = latest_val_log[f1]

            if latest_val_log[precision] > self.now_best_val_log[precision]:
                self.now_best_val_log[precision] = latest_val_log[precision]

            if latest_val_log[recall] > self.now_best_val_log[recall]:
                self.now_best_val_log[recall] = latest_val_log[recall]


            return False

    def __call__(self, val_logs=None, epoch=0):

        if not self.always_save:
            if self.is_better(latest_val_log=val_logs):
                file_path = os.path.join(self.save_model_path, str(epoch))
                self.log.debug('NEW Best!! Save model ckpt to %s' % file_path)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                    file = os.path.join(file_path, 'ckpt.pth')
                    torch.save({
                        'val_logs':self.now_best_val_log,
                        'epoch':epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, file)
                else:
                    RuntimeError('Model file has exist!')
            self.log.debug('So far best: ' + ' - '.join(f'{k}: {v}' for k, v in self.now_best_val_log.items()))
        else:
            file_path = os.path.join(self.save_model_path, str(epoch))
            self.log.debug('Save model ckpt to %s' % file_path)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                file = os.path.join(file_path, 'ckpt.pth')
                torch.save({
                    'val_logs':self.now_best_val_log,
                    'epoch':epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, file)
            else:
                RuntimeError('Model file has exist!')




class EvaluateMetrics():

    def __init__(
        self,
        metric,
        generator=None,
        save_model_callback = None,
        once_every: int = 1,
        verbose=1,
        mode='max',
        log = None
    ):
        """Initializer."""
        self.log = log
        self.metrics = metric
        self._dev_x, self.dev_y = generator.get_all()
        self._valid_steps = once_every
        self.save_model_callback = save_model_callback
        self._verbose = verbose
        self.mode = mode

    def eval_metric_on_data_frame(
        cls,
        metric: BaseMetric,
        id_left: typing.Union[list, np.array],
        y: typing.Union[list, np.array],
        y_pred: typing.Union[list, np.array]
    ):
        eval_df = pd.DataFrame(data={
            'id': id_left,
            'true': y.squeeze(),
            'pred': y_pred.squeeze()
        })
        assert isinstance(metric, BaseMetric)

        eval_df = eval_df.groupby('id').apply(f)

        eval_df['id'] = np.ones_like(eval_df['id'])

        val = eval_df.groupby(by='id').apply(
            lambda df: metric(df['true'].values, df['pred'].values)
        ).mean()
        return val

    def eval(self, epoch: int, pred = None, step = -1, summary_writer = None, mode = None):
        """
        Called at the end of en epoch.

        :param epoch: integer, index of epoch.
        :param logs: dictionary of logs.
        :return: dictionary of logs.
        """
        if (epoch + 1) % self._valid_steps == 0:
            val_logs = {}
            assert pred.shape[0] == self.dev_y.shape[0]

            dev_x={
                'id_left':list(self._dev_x['id_left']) ,
                'id_right':list(self._dev_x['id_right']) ,
                'label':list(self.dev_y),
                'pred':list(np.squeeze(pred))
            }
            dev_data = pd.DataFrame(dev_x)

            if self.mode == 'max':
                dev_data = dev_data.iloc[dev_data.groupby(['id_left', 'id_right']).apply(lambda x:x['pred'].idxmax())]
            elif self.mode == 'mean':
                pred_mean = dev_data.groupby(['id_left', 'id_right'])['pred'].transform('mean')
                del dev_data['pred']
                dev_data['pred'] = pred_mean
                dev_data = dev_data.iloc[dev_data.groupby(['id_left', 'id_right']).apply(lambda x: x['pred'].idxmax())]
            else:
                RuntimeError('Eval mode error!!')

            x = {
                'id_left': dev_data['id_left'].values,
                'id_right': dev_data['id_right'].values,
            }

            y = dev_data['label'].values
            y_pred = dev_data['pred'].values
            matchzoo_metrics = self.metrics
            for metric in matchzoo_metrics:
                val_logs[metric] = self.eval_metric_on_data_frame(metric, x['id_left'], y, y_pred)

            if self._verbose:
                self.log.debug('Validation: ' + ' - '.join(f'{k}: {v}' for k, v in val_logs.items()))

            if mode and summary_writer and step != -1:
                for k, v in val_logs.items():
                    summary_writer.add_scalar(mode + '/' + str(k) , v, step)
                    summary_writer.flush()

            if self.save_model_callback:
                self.save_model_callback(val_logs, epoch)


def f(group):
    index = group['pred'].idxmax()
    group.loc[index, 'pred'] = -1000.0
    group.loc[group['pred'] != -1000, 'pred' ] = 0.0
    group.loc[group['pred'] == -1000, 'pred'] = 1.0
    return group

if __name__ == '__main__':
    eval_df = pd.DataFrame(data={
        'id': [1, 1, 1, 2, 2, 2],
        'true': [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'pred': [0.3, 0.76, 0.54, 0.18, 0.99, 0.78]
    })

    mc = F1()

    print(eval_df)

    eval_df = eval_df.groupby('id').apply(f)

    eval_df['id'] = np.ones_like(eval_df['id'])

    score = eval_df.groupby(by='id').apply(
        lambda df: mc(df['true'].values, df['pred'].values)
    ).mean()


    # score = mc(eval_df['true'].values, eval_df['pred'].values)

    print(eval_df)

    print(score)



