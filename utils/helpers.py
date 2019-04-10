#!/usr/bin/env python
import torch
import time

def train_model(model, optimizer, nb_epochs, train_input, train_target ,mini_batch_size):

    start = time.time()
    for e in range(0,nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            target = train_target.narrow(0, b, mini_batch_size)
            loss = model.criterion(output, target)
            model.zero_grad()
            loss.backward()
            optimizer.step()
    end = time.time()

    training_time = end-start

    return training_time

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        pred = model.predict(data_input.narrow(0, b, mini_batch_size))
        for k in range(mini_batch_size):
            if data_target.data[b + k] != pred[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

def update_target_type(model, data_target, test_target):
    type_ = model.target_type
    return data_target.type(type_), test_target.type(type_)