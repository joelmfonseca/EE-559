#!/usr/bin/env python
import torch

from utils.data import batch_iter
from math import ceil

def compute_accuracy(yHat, y, batch_size): 
    n_samples = len(yHat)
    n_batches = ceil(n_samples / batch_size)
    n_errors = 0
    for m, n in batch_iter(yHat, y, batch_size, num_batches=n_batches, shuffle=False):
        for i in range(len(n)):
            if m[i] != n[i]:
                n_errors += 1
    
    return torch.Tensor([1 - n_errors / n_samples])

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        pred = model.predict(data_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if data_target.data[b + k] != pred[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors