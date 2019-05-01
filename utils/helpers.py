#!/usr/bin/env python
import sys
import time
import torch

from utils.objects import History
from utils.metrics import compute_accuracy, compute_nb_errors
from utils.data import batch_iter, split_data

def h_mean(histories):
    m_history = History()
    m_history.mean = True
    n = len(histories)
    
    for metric in histories[0].history:
        acc = torch.Tensor([0]*histories[0].epochs)
        for h in histories:
            acc += torch.Tensor(h.history[metric])
        m_history.history[metric] = acc / n
    return m_history   

def h_stats(histories):
    stats = {}
    for metric in histories[0].history:
        stats[metric] = {}
        min_, max_ = sys.maxsize, -sys.maxsize+1
        for h in histories:
            curr_min, curr_max = h.history[metric].min().item(), h.history[metric].max().item()
            if curr_min < min_:
                min_ = curr_min
            if curr_max > max_:
                max_ = curr_max
        stats[metric]['min'], stats[metric]['max'] = min_, max_

    return stats

def update_history(history, model, tx, y, mini_batch_size):
    # Split data between training and validation
    (x_train, y_train), (x_test, y_test) = split_data(y, tx)
    # Evalutation mode
    model.eval()
        
    # Training
    yHat_train = model(x_train)
    train_loss = torch.Tensor([model.criterion(yHat_train, y_train).item()])
    #train_acc = compute_accuracy(yHat_train, y_train, mini_batch_size)
    train_acc = torch.Tensor([1 - compute_nb_errors(model, x_train, y_train, mini_batch_size) / x_train.size(0)])
    
    history.history['acc'] = torch.cat((history.history['acc'], train_acc), 0)
    history.history['loss'] = torch.cat((history.history['loss'], train_loss), 0)
        
    # Validation
    yHat_test = model(x_test)
    #print(yHat_test.shape, y_test.shape)
    #print(yHat_test.narrow(0, 0, 10))
    #print(y_test.narrow(0, 0, 10))
    #sys.exit()
        
    test_loss = torch.Tensor([model.criterion(yHat_test, y_test).item()])
    yHat_test = model.predict(x_test)
    #test_acc = compute_accuracy(yHat_test, y_test, mini_batch_size)
    test_acc = torch.Tensor([1 - compute_nb_errors(model, x_test, y_test, mini_batch_size) / x_test.size(0)])
    history.history['val_acc'] = torch.cat((history.history['val_acc'], test_acc), 0)
    history.history['val_loss'] = torch.cat((history.history['val_loss'], test_loss), 0)
    
    return history

# TODO: Argument for the way splitting is performed
def train_model(model, optimizer, n_epochs, tx, y, batch_size):
    # Split data between training and validation
    (x_train, y_train), (x_test, y_test) = split_data(y, tx)
    # Get number of training examples
    n_samples = x_train.size(0)
    # Save training history
    history = History()
    
    start = time.time()
    for e in range(0, n_epochs):
        model.train()
        #e_loss = torch.Tensor([.0])
        for b in range(0, n_samples, batch_size):
            curr_x_train = x_train.narrow(0, b, batch_size)
            curr_y_train = y_train.narrow(0, b, batch_size)
            
            # Forward pass
            pred = model(curr_x_train)
            # Backward pass
            loss = model.criterion(pred, curr_y_train)
            # zero the parameter gradients
            model.zero_grad()
     
            loss.backward()                        
            # Update weights
            optimizer.step()
            # Update loss
            #e_loss += loss.item()
        
        update_history(history, model, tx, y, batch_size)    
    end = time.time()
    
    history.training_time = end-start
    history.epochs = n_epochs
    history.model['name'] = model.__class__.__name__
    history.model['optim'] = optimizer
    history.model['criterion'] = model.criterion.__class__.__name__
    return history