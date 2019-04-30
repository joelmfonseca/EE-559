#!/usr/bin/env python
import torch
import visdom

class History():
    def __init__(self):
        self.training_time = None
        self.history = {'val_loss': torch.Tensor([]), 'val_acc': torch.Tensor([]), 
                        'loss': torch.Tensor([]), 'acc': torch.Tensor([])}
        self.mean = False
        self.epochs = None
        
    def __stat__(self):
        stat = {'val_loss': {}, 'val_acc': {}, 'loss': {}, 'acc': {}}
        
        for metric in self.history:
            stat[metric]['min'] = self.history[metric].min()
            stat[metric]['max'] = self.history[metric].max()

        return stat

class Visualization():
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None
        self.accuracy_win = None
        
    def plot_loss(self, loss, epoch, opt_name):
        self.loss_win = self.vis.line(
            [loss],
            [epoch],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='epoch',
                ylabel=opt_name,
                title='Loss (mean per 10 steps)',
            )
        )
        
    def plot_accuracy(self, train_accuracy, validation_accuracy, epoch, opt_name):
        self.accuracy_win = self.vis.line(
            [train_accuracy],
            [epoch],
            win=self.accuracy_win,
            update='append' if self.accuracy_win else None,
            name='train',
            opts=dict(
                xlabel='epoch',
                ylabel=opt_name,
                title='% Accuracy (mean per 10 steps)',
            )
        )
        
        self.accuracy_win = self.vis.line(
            [validation_accuracy],
            [epoch],
            win=self.accuracy_win,
            update='append' if self.accuracy_win else None,
            name='validation',
            opts=dict(
                xlabel='epoch',
                ylabel=opt_name,
                title='% Accuracy (mean per 10 steps)',
            )
        )