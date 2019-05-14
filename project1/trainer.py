import time
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

import dlc_practical_prologue as prologue


def train_model(model, optimizer, nb_epochs, train_input, train_target, test_input, test_target, mini_batch_size, graph = False):

    test  = []
    train = []
    start = time.time()
    for e in range(0,nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            target = train_target.narrow(0, b, mini_batch_size)
            loss = model.criterion(output, target)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if graph:
            test.append(100 - compute_nb_errors(model, test_input, test_target, mini_batch_size) / train_input.size(0) * 100)
            train.append(100 - compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100)
    
    end = time.time()

    training_time = end-start

    if graph:
        return test,train
    else :
        return training_time


def train_model_aux(model, optimizer, nb_epochs, train_input, train_target, train_class, test_input, test_target, mini_batch_size, \
                   alpha_1, alpha_2, alpha_3, graph = False):

    test = []
    train = []
    start = time.time()
    for e in range(0,nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output_class_0, output_class_1, output_bin = model(train_input.narrow(0, b, mini_batch_size))
            target_class = train_class.narrow(0, b, mini_batch_size)
            target_bin = train_target.narrow(0, b, mini_batch_size)
            
            target_class_0 = torch.Tensor([x[0].item() for x in target_class]).type(torch.LongTensor)
            target_class_1 = torch.Tensor([x[1].item() for x in target_class]).type(torch.LongTensor)
            
            l1 = model.criterion(output_class_0, target_class_0)
            l2 = model.criterion(output_class_1, target_class_1)
            l3 = model.criterion(output_bin,target_bin)
            
            loss = alpha_1*l1 + alpha_2*l2 + alpha_3*l3
        
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if graph:
            test.append(100 - compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100)
            train.append(100 - compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100)

    end = time.time()
    training_time = end-start

    if graph:
        return test,train
    else:
        return training_time

def train_model_aux_bin(model, optimizer, nb_epochs, train_input, train_target, train_class, test_input, test_target, mini_batch_size, \
                   alpha_1, alpha_2, alpha_3, graph = False):

    test = []
    train = []
    start = time.time()
    for e in range(0,nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output_class_0, output_class_1, output_bin = model(train_input.narrow(0, b, mini_batch_size))
            target_class = train_class.narrow(0, b, mini_batch_size)
            target_bin = train_target.narrow(0, b, mini_batch_size)
            
            target_class_0 = torch.Tensor([x[0].item() for x in target_class]).type(torch.LongTensor)
            target_class_1 = torch.Tensor([x[1].item() for x in target_class]).type(torch.LongTensor)
            
            l1 = model.criterion[0](output_class_0, target_class_0)
            l2 = model.criterion[0](output_class_1, target_class_1)
            l3 = model.criterion[1](output_bin.view(mini_batch_size),target_bin)
            
            loss = alpha_1*l1 + alpha_2*l2 + alpha_3*l3
        
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if graph:
            test.append(100 - compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100)
            train.append(100 - compute_nb_errors(model, train_input, train_target, mini_batch_size) / test_input.size(0) * 100)
        
    end = time.time()
    training_time = end-start

    if graph:
        return test, train
    else:
        return training_time


