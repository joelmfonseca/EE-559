#!/usr/bin/env python

import torch
import time

import dlc_practical_prologue as prologue

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from utils import *

# create different models
#Vanilla model
class Net(nn.Module):
    def __init__(self,nb_hidden = 100):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

    def predict(self,x):
        pred = torch.tensor([self.compare_pair(i) for i in x])
        return pred

    def compare_pair(self, x):
        a = x[0]
        b = x[1]
        val_a = self.forward(a.view(1,1,14,14)).max(1)[1].item()
        val_b = self.forward(b.view(1,1,14,14)).max(1)[1].item()

        if(val_a > val_b):
            return 0
        else:
            return 1


class Net1(nn.Module):
    def __init__(self, nb_hidden=100):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

    def predict(self, x):
        return torch.max(self.forward(x), 1)[1]

class Net2(nn.Module):
    def __init__(self, nb_hidden=100):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.target_type = torch.FloatTensor

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x.view(-1)

    def predict(self, x):
        return torch.sigmoid(self.forward(x)).round()



def train_model_vanilla(model, optimizer, nb_epochs, train_input, train_target ,mini_batch_size):
    
    start = time.time()
    for e in range(0, nb_epochs):
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            target = train_target.narrow(0, b, mini_batch_size)
            loss = model.criterion(output,target.max(1)[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    end = time.time()

    return end-start
    

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

if __name__ == '__main__':

    # load the data
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(nb=1000)

    # normalize it
    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    train_input_vanilla, train_target_vanilla = prepare_vanilla(train_input, train_classes)
    train_input_vanilla, train_target_vanilla = Variable(train_input_vanilla), Variable(train_target_vanilla)
    
    train_input, train_target = Variable(train_input), Variable(train_target)
    test_input, test_target = Variable(test_input), Variable(test_target)

  

    # check different configurations
    NB_EPOCHS = 25
    MINI_BATCH_SIZE = 100
    models = [Net, Net1]
    optimizers = [optim.SGD, optim.Adam]
    learning_rates = [1e-1, 1e-2, 1e-3]

    for m in models:

        model = m()
            


            
        train_target, test_target = update_target_type(model, train_target, test_target)

        

        for optimizer in optimizers:
            for learning_rate in learning_rates:
                
                



                if(model.__class__.__name__ == 'Net'):
                    training_time = train_model_vanilla(model, optimizer(model.parameters(), lr=learning_rate), NB_EPOCHS, \
                        train_input_vanilla, train_target_vanilla, MINI_BATCH_SIZE)

                    
                    print('model: {:>5}, criterion: {:>10}, optimizer: {:>10}, learning rate: {:6}, num epochs: {:3}, '
                        'mini batch size: {:3}, training time: {:5.2f}, train error: {:5.2f}%, test error: {:5.2f}%'.format(
                        model.__class__.__name__,
                        model.criterion.__class__.__name__,
                        optimizer.__name__,
                        learning_rate,
                        NB_EPOCHS,
                        MINI_BATCH_SIZE,
                        training_time,
                        compute_nb_errors(model, train_input, train_target, MINI_BATCH_SIZE) / train_input.size(0) * 100,
                        compute_nb_errors(model, test_input, test_target, MINI_BATCH_SIZE) / test_input.size(0) * 100
                        )
                )
                    
            
                else:
                    training_time = train_model(model, optimizer(model.parameters(), lr=learning_rate), NB_EPOCHS, \
                        train_input, train_target, MINI_BATCH_SIZE)
                    print('model: {:>5}, criterion: {:>10}, optimizer: {:>10}, learning rate: {:6}, num epochs: {:3}, '
                        'mini batch size: {:3}, training time: {:5.2f}, train error: {:5.2f}%, test error: {:5.2f}%'.format(
                        model.__class__.__name__,
                        model.criterion.__class__.__name__,
                        optimizer.__name__,
                        learning_rate,
                        NB_EPOCHS,
                        MINI_BATCH_SIZE,
                        training_time,
                        #compute_nb_errors(model, train_input, train_target, MINI_BATCH_SIZE) / train_input.size(0) * 100,
                        training_time,
                        compute_nb_errors(model, test_input, test_target, MINI_BATCH_SIZE) / test_input.size(0) * 100
                        )
                )