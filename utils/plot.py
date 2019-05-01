#!/usr/bin/env python
import torch

from math import exp
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from utils.helpers import h_stats, h_mean


def prepare_standardplot(title, xlabel, history):
    stats = h_stats([history])
    loss_min = min(stats['loss']['min'], stats['val_loss']['min'])
    loss_max = max(stats['loss']['max'], stats['val_loss']['max'])
    acc_min = min(stats['acc']['min'], stats['val_acc']['min'])
    ax1_offset = (loss_max-loss_min)*.1
    ax2_offset = (1-acc_min)*.1
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)    
    ax1.set_ylabel(history.model['criterion'])
    ax1.set_xlabel(xlabel)    
    #ax1.set_yscale('log')
    ax1.set_ylim(loss_min, loss_max+ax1_offset)
    ax1.set_xlim(0, history.epochs)        
    ax2.set_ylabel('accuracy [% correct]')
    ax2.set_xlabel(xlabel)
    ax2.set_ylim(acc_min-ax2_offset, 1+ax2_offset)
    ax2.set_xlim(0, history.epochs)
    
    return fig, ax1, ax2

def finalize_standardplot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels, loc='lower right')
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

def plot_history(history, title):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch', history)
    ax1.plot(history.history['loss'].tolist(), label = "training")
    ax1.plot(history.history['val_loss'].tolist(), label = "validation")
    ax2.plot(history.history['acc'].tolist(), label = "training")
    ax2.plot(history.history['val_acc'].tolist(), label = "validation")
    finalize_standardplot(fig, ax1, ax2)
    return fig

def comparison_plot(history_1, history_2, label1, label2, title):
    fig, ax1, ax2 = prepare_standardplot(title, "epochs")
    ax1.plot(history['loss'], label=label1 + ' training')
    ax1.plot(history['val_loss'], label=label1 + ' validation')
    ax1.plot(history['loss'], label=label2 + ' training')
    ax1.plot(history['val_loss'], label=label2 + ' validation')
    ax2.plot(history['acc'], label=label1 + ' training')
    ax2.plot(history['val_acc'], label=label1 + ' validation')
    ax2.plot(history['acc'], label=label2 + ' training')
    ax2.plot(history['val_acc'], label=label2 + ' validation')
    finalize_standardplot(fig, ax1, ax2)
    return fig

def plot_histories(histories, title):
    fig, ax1, ax2 = prepare_standardplot(title, 'epoch', histories[0].epochs, h_stats(histories))
    xs = torch.arange(0, epochs).tolist()
    alpha = 1/8
    # Lines
    loss, acc, val_loss, val_acc = [], [], [], []
    for history in histories + [h_mean(histories)]:
        # Create lines
        a = list(zip(xs, history.history['loss'].tolist()))
        b = list(zip(xs, history.history['val_loss'].tolist()))
        c = list(zip(xs, history.history['acc'].tolist()))
        d = list(zip(xs, history.history['val_acc'].tolist()))
        if history.mean:
            m_alpha = alpha*5
            ax1.add_collection(LineCollection([a], label = "training", color='orange', alpha=m_alpha))
            ax1.add_collection(LineCollection([b], label = "validation", color='lightskyblue', alpha=m_alpha))
            ax2.add_collection(LineCollection([c], label = "training", color='orange', alpha=m_alpha))
            ax2.add_collection(LineCollection([d], label = "validation", color='lightskyblue', alpha=m_alpha))
        else:            
            # Update
            loss.append(a), val_loss.append(b)
            acc.append(c), val_acc.append(d)
    
    ax1.add_collection(LineCollection(loss, color='orange', alpha=alpha))
    ax1.add_collection(LineCollection(val_loss, color='lightskyblue', alpha=alpha))
    ax2.add_collection(LineCollection(acc, color='orange', alpha=alpha))
    ax2.add_collection(LineCollection(val_acc, color='lightskyblue', alpha=alpha))
    
    finalize_standardplot(fig, ax1, ax2)
    return fig