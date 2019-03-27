#!/usr/bin/env python

import torch
import time

import dlc_practical_prologue as prologue

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

# create different models
class Net1(nn.Module):
    def __init__(self, nb_hidden=100):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

def train_model(model, criterion, optimizer, nb_epochs, train_input, train_target, mini_batch_size):

    start = time.time()
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
    end = time.time()

    training_time = end-start

    return training_time

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

if __name__ == '__main__':

    # load the data
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(nb=1000)

    # normalize it
    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    train_input, train_target = Variable(train_input), Variable(train_target)
    test_input, test_target = Variable(test_input), Variable(test_target)

    # check different configurations
    NB_EPOCHS = 25
    MINI_BATCH_SIZE = 100
    models = [Net1]
    criterions = [nn.CrossEntropyLoss()]
    optimizers = [optim.SGD, optim.Adam]
    learning_rates = [1e-1, 1e-2, 1e-3]

    for m in models:
        for criterion in criterions:
            for optimizer in optimizers:
                for learning_rate in learning_rates:

                    model = m()
                    training_time = train_model(model, criterion, \
                        optimizer(model.parameters(), lr=learning_rate), NB_EPOCHS, \
                            train_input, train_target, MINI_BATCH_SIZE)
                    
                    print('model: {:>5}, criterion: {:>10}, optimizer: {:>10}, learning rate: {:6}, num epochs: {:3}, '
                          'mini batch size: {:3}, training time: {:5.2f}, train error: {:5.2f}%, test error: {:5.2f}%'.format(
                            model.__class__.__name__,
                            criterion.__class__.__name__,
                            optimizer.__name__,
                            learning_rate,
                            NB_EPOCHS,
                            MINI_BATCH_SIZE,
                            training_time,
                            compute_nb_errors(model, train_input, train_target, MINI_BATCH_SIZE) / train_input.size(0) * 100,
                            compute_nb_errors(model, test_input, test_target, MINI_BATCH_SIZE) / test_input.size(0) * 100
                            )
                    )