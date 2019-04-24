#!/usr/bin/env python

import torch
import time

import dlc_practical_prologue as prologue

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from utils import *

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

class Net3(nn.Module):
    def __init__(self,nb_hidden = 100):
        super(Net3, self).__init__()
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

        def compare_pair(x):
            a = x[0]
            b = x[1]
            val_a = self.forward(a.view(1,1,14,14)).max(1)[1].item()
            val_b = self.forward(b.view(1,1,14,14)).max(1)[1].item()

            if(val_a > val_b):
                return 0
            else:
                return 1

        pred = torch.tensor([compare_pair(i) for i in x])
        
        return pred

class NetSharing1(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetSharing1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x_0 = x[:,0,:,:].view(-1,1,14,14)
        x_1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv1(x_0), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)

        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        return torch.max(self.forward(x), 1)[1]

class NetSharing2(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetSharing2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.target_type = torch.FloatTensor

    def forward(self, x):
        x_0 = x[:,0,:,:].view(-1,1,14,14)
        x_1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv1(x_0), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)

        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1)

    def predict(self, x):
        return torch.sigmoid(self.forward(x)).round()

class NetSharing3(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetSharing3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x_0 = x[:,0,:,:].view(-1,1,14,14)
        x_1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv1(x_0), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)

        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        return torch.max(self.forward(x), 1)[1]

class NetAuxiliary1(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetAuxiliary1, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(10, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        y = self.fc3(F.relu(x))
        return x, y

    def predict(self, x):
        return torch.max(self.forward(x)[1], 1)[1]

class NetAuxiliary2(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetAuxiliary2, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(nb_hidden, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        z = self.fc2(x)
        y = self.fc3(x)
        return z, y

    def predict(self, x):
        return torch.max(self.forward(x)[1], 1)[1]

class NetAuxiliary3(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetAuxiliary3, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.fc1 = nn.Sequential(nn.Linear(1152, 500), nn.ReLU, nn.Linear(500, 100), nn.ReLU, nn.Linear(100, 10))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc2 = nn.Linear(256, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        z = self.fc1(x.view(-1, 1152))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc2(x.view(-1, 256)))
        y = self.fc3(x)
        return z, y

    def predict(self, x):
        return torch.max(self.forward(x)[1], 1)[1]
    
def train_model_auxiliary(model, optimizer, nb_epochs, train_input, train_target_bin,
                            train_target_class, mini_batch_size):

    start = time.time()
    for e in range(0,nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output_class, output_bin = model(train_input.narrow(0, b, mini_batch_size))

            target_bin = train_target_bin.narrow(0, b, mini_batch_size)
            target_class = train_target_class.narrow(0, b, mini_batch_size)

            loss_bin = model.criterion(output_bin, target_bin)
            loss_class = model.criterion(output_class, target_class)

            lambda_ = 0
            loss = loss_bin + lambda_*loss_class

            model.zero_grad()
            loss.backward()
            optimizer.step()
    end = time.time()

    training_time = end-start

    return training_time

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

    train_input_Net3 = train_input.view(-1, 1, 14, 14)
    train_target_Net3 = train_classes.flatten()
    
    train_input, train_target, train_classes = Variable(train_input), Variable(train_target), Variable(train_classes)
    test_input, test_target = Variable(test_input), Variable(test_target)

    train_input_Net3 = train_input.view(-1, 1, 14, 14)
    train_target_Net3 = train_classes.flatten()

    # test different configurations
    NB_EPOCHS = 25
    MINI_BATCH_SIZE = 100
    # models = [Net1, Net2, Net3, NetSharing1, NetSharing2]
    models = [NetAuxiliary3]
    optimizers = [optim.SGD, optim.Adam]
    learning_rates = [1e-1, 1e-2, 1e-3]

    for m in models:

        model = m()
        train_target, test_target = update_target_type(model, train_target, test_target)

        for optimizer in optimizers:
            for learning_rate in learning_rates:
                  
                if model.__class__.__name__ == 'Net3':
                    training_time = train_model(model, optimizer(model.parameters(), lr=learning_rate), NB_EPOCHS, \
                        train_input_Net3, train_target_Net3, MINI_BATCH_SIZE)
                
                elif 'Auxiliary' in model.__class__.__name__:
                    training_time = train_model_auxiliary(model, optimizer(model.parameters(), lr=learning_rate), NB_EPOCHS, \
                        train_input, train_target, train_target_Net3, MINI_BATCH_SIZE)
                
                else:
                    training_time = train_model(model, optimizer(model.parameters(), lr=learning_rate), NB_EPOCHS, \
                        train_input, train_target, MINI_BATCH_SIZE)

                print('model: {:>13}, criterion: {:>10}, optimizer: {:>10}, learning rate: {:6}, num epochs: {:3}, '
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
