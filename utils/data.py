#!/usr/bin/env python
import torch
from torch.autograd import Variable

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    torch.manual_seed(0)
    data_size = len(y)

    if shuffle:
        shuffle_indices = torch.randperm(data_size).tolist()
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def split_data(y, tx, ratio=0.2):
    torch.manual_seed(0)
    data_size = len(y)

    shuffle_indices = torch.randperm(data_size).tolist()
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    
    cut = int(data_size*ratio)
    return (shuffled_tx[cut:], shuffled_y[cut:]), (shuffled_tx[:cut], shuffled_y[:cut])

def data_format(model, data):
    # Get standard data
    (x_train, y_train, c_train), (x_test, y_test, c_test) = data
    # Standardize input
    mean, std = x_train.mean(), x_train.std()
    x_train.sub_(mean).div_(std), x_test.sub_(mean).div_(std)
    # Format output
    y_train, y_test = y_train.type(model.target_type), y_test.type(model.target_type)
    # Format classes
    c_train, c_test = Variable(c_train), c_test
    # Format input
    if model.__class__.__name__ == 'Net3':
        x_train, x_test = x_train.view(-1, 1, 14, 14), x_classes.flatten()
    else:
        x_train, x_test = Variable(x_train), Variable(x_test)
    
    return (x_train, y_train, c_train), (x_test, y_test, c_test)

def update_target_type(model, y_train, y_test):
    type_ = model.target_type
    return y_train.type(type_), y_test.type(type_)