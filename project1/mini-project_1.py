#!/usr/bin/env python
import time
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

import dlc_practical_prologue as prologue

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

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

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

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

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

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

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

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

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

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

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

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

class NetAux1(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetAux1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(20,2)

        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x_0 = x[:,0,:,:].view(-1,1,14,14)
        x_1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv1(x_0), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1 = F.relu(self.fc1(x1)) 
        #original en dessous
        x1 = self.fc2(x1)
        #test en dessous
        #x1 = F.relu(self.fc2(x1))

        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        #original en dessous
        x2 = self.fc2(x2)
        #test en dessous
        #x2 = F.relu(self.fc2(x2))

        x = torch.cat((x1,x2),1)
        x = self.fc3(x)
        
        return x1,x2,x

    def predict(self, x):
        
        _, _ , pred = self.forward(x)
        pred = torch.max(pred,1)[1]
        return pred

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

class NetAux2(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetAux2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(20,nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.target_type = torch.LongTensor

    def forward(self, x):
        x_0 = x[:,0,:,:].view(-1,1,14,14)
        x_1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv1(x_0), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1 = F.relu(self.fc1(x1)) 
        #original en dessous
        x1 = self.fc2(x1)
        #test en dessous
        #x1 = F.relu(self.fc2(x1))
        
        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        #original en dessous
        x2 = self.fc2(x2)
        #test en dessous
        #x2 = F.relu(self.fc2(x2))

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = F.relu(self.fc4(x))
        
        return x1,x2,x

    def predict(self, x):
        
        _, _ , pred = self.forward(x)
        pred = torch.max(pred,1)[1]
        return pred

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

class NetAux3(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetAux3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(20,nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 1)

        self.criterion = [nn.CrossEntropyLoss(),nn.BCEWithLogitsLoss()]
        self.target_type = torch.FloatTensor

    def forward(self, x):
        x_0 = x[:,0,:,:].view(-1,1,14,14)
        x_1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv1(x_0), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1 = F.relu(self.fc1(x1)) 
        #original en dessous
        x1 = self.fc2(x1)
        #test en dessous
        #x1 = F.relu(self.fc2(x1))
        
        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        #original en dessous
        x2 = self.fc2(x2)
        #test en dessous
        #x2 = F.relu(self.fc2(x2))

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = F.relu(self.fc4(x))
        
        return x1,x2,x

    def predict(self, x):
        _, _ , pred = self.forward(x)
        return torch.sigmoid(pred).round()
        
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

class NetAux4(nn.Module):
    def __init__(self, nb_hidden=100):
        super(NetAux4, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.fc3 = nn.Linear(20,nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 256)
        self.fc5 = nn.Linear(256,1)

        self.criterion = [nn.CrossEntropyLoss(),nn.BCEWithLogitsLoss()]
        self.target_type = torch.FloatTensor

    def forward(self, x):
        x_0 = x[:,0,:,:].view(-1,1,14,14)
        x_1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(F.max_pool2d(self.conv1(x_0), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, 256)
        x1 = F.relu(self.fc1(x1)) 
        #original en dessous
        x1 = self.fc2(x1)
        #test en dessous
        #x1 = F.relu(self.fc2(x1))
        
        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        #original en dessous
        x2 = self.fc2(x2)
        #test en dessous
        #x2 = F.relu(self.fc2(x2))

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.fc5(x)

        return x1,x2,x

    def predict(self, x):
        
        _, _ , pred = self.forward(x)
        return torch.sigmoid(pred).round()

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

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

def plot_model_comparison(train_input, train_target, train_class, test_input, test_target, optimizer, learning_rate, nb_epochs, mini_batch_size):
    model_raw = Net1()
    train_target, test_target = update_target_type(model_raw, train_target, test_target)
    test_raw, train_raw = train_model(model_raw, optimizer(model_raw.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    model_weight = NetSharing1()
    train_target, test_target = update_target_type(model_weight, train_target, test_target)
    test_weight, train_weight = train_model(model_weight, optimizer(model_weight.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    model_aux = NetAux4()
    train_target, test_target = update_target_type(model_aux, train_target, test_target)
    test_aux, train_aux = train_model_aux_bin(model_aux, optimizer(model_aux.parameters(), lr=learning_rate, momentum = 0.7), nb_epochs, \
                         train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5, True)

    plt.plot(test_raw, label = 'Standard model')
    plt.plot(test_weight, label = 'Weight sharing model')
    plt.plot(test_aux, label = 'Auxiliary model')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on the test set (%)')
    plt.legend(loc = 'best')
    plt.title('Comparison between the three models')
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300)
    plt.show()

def plot_netsharing_comparison(train_input, train_target, test_input, test_target, optimizer, learning_rate, nb_epochs, mini_batch_size):
    model_weight_1 = NetSharing1()
    train_target, test_target = update_target_type(model_weight_1, train_target, test_target)
    test_weight_1, train_weight_1 = train_model(model_weight_1, optimizer(model_weight_1.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    model_weight_2 = NetSharing2()
    train_target, test_target = update_target_type(model_weight_2, train_target, test_target)
    test_weight_2, train_weight_2 = train_model(model_weight_2, optimizer(model_weight_2.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    model_weight_3 = NetSharing3()
    train_target, test_target = update_target_type(model_weight_3, train_target, test_target)
    test_weight_3, train_weight_3 = train_model(model_weight_3, optimizer(model_weight_3.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size, True)

    plt.plot(test_weight_1,label = 'Weight sharing model 1')
    plt.plot(test_weight_2, label = 'Weight sharing model 2')
    plt.plot(test_weight_3, label = 'Weight sharing model 3')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on the test set (%)')
    plt.legend(loc = 'best')
    plt.title('Comparison between the three weight sharing models')
    plt.tight_layout()
    plt.savefig('figures/weight_sharing_comparison.png', dpi=300)
    plt.show()

def grid_search(models, optimizers, learning_rates, train_input, train_target, train_class, test_input, test_target, nb_epochs, mini_batch_size):
    for m in models:
        for optimizer in optimizers:
            for learning_rate in learning_rates:
                
                model = m()
                model.apply(model.weight_init)
                train_target, test_target = update_target_type(model, train_target, test_target)
                
                if model.__class__.__name__ == 'Net3':
                    train_input_Net3 = train_input.view(-1, 1, 14, 14)
                    train_target_Net3 = train_class.flatten()

                if model.__class__.__name__ == 'Net3':
                    training_time = train_model(model, optimizer(model.parameters(), lr=learning_rate), nb_epochs, \
                                                    train_input_Net3, train_target_Net3, mini_batch_size)

                elif model.__class__.__name__ == 'NetAux1' or model.__class__.__name__ == "NetAux2":
                    alphas = [0.8, 0.8, 1]
                    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.85)
                    training_time = train_model_aux(model, optimizer(model.parameters(), lr=learning_rate), nb_epochs, \
                                                        train_input, train_target, train_class, mini_batch_size, 0.2, 0.8, 1.5)
                
                elif model.__class__.__name__ == 'NetAux3' or model.__class__.__name__ == "NetAux4":
                    alphas = [0.8, 0.8, 1]
                    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.85)
                    if optimizer == optim.SGD:
                        training_time = train_model_aux_bin(model, optimizer(model.parameters(), lr=learning_rate, momentum = 0.7), nb_epochs, \
                                                                train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5)
                    elif optimizer == optim.RMSprop:
                        training_time = train_model_aux_bin(model, optimizer(model.parameters(), lr=learning_rate, alpha = 0.9), nb_epochs, \
                                                                train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5)
                    else:
                        training_time = train_model_aux_bin(model, optimizer(model.parameters(), lr=learning_rate, betas=(0.9,0.99)), nb_epochs, \
                                                                train_input, train_target, train_class, test_input, test_target, mini_batch_size, 0.2, 0.8, 1.5)
                else:
                    training_time = train_model(model, optimizer(model.parameters(), lr=learning_rate), nb_epochs, \
                        train_input, train_target, test_input, test_target, mini_batch_size)
                            
                print('model: {:>13}, criterion: {:>10}, optimizer: {:>10}, learning rate: {:6}, num epochs: {:3}, '
                    'mini batch size: {:3}, training time: {:5.2f}, train error: {:5.2f}%, test error: {:5.2f}%'.format(
                    model.__class__.__name__,
                    model.criterion.__class__.__name__,
                    optimizer.__name__,
                    learning_rate,
                    nb_epochs,
                    mini_batch_size,
                    training_time,
                    compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100,
                    compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
                    )
                )

if __name__ == '__main__':

    # load the data
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        prologue.generate_pair_sets(nb=1000)

    # normalize it
    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    train_input, train_target, train_class = Variable(train_input), Variable(train_target), Variable(train_classes)
    test_input, test_target = Variable(test_input), Variable(test_target)

    # test different configurations
    nb_epochs = 25
    mini_batch_size = 40
    #models = [Net1, Net2, Net3, NetSharing1, NetSharing2, NetSharing3, NetAuxiliary1]
    #models = [NetSharing1, NetSharing2, NetSharing3, Netaux]
    #models = [NetAuxiliary1, NetAuxiliary2]
    #models= [Net1, Net2, NetSharing1, NetSharing2, NetSharing3 ,NetAux1, NetAux2, NetAux3, NetAux4, Net3]
    models = [NetAux4]
    optimizers = [optim.SGD]
    #optimizers = [optim.SGD, optim.Adam, optim.RMSprop]
    learning_rates = [1e-1]
    #learning_rates = [1e-1, 1e-2, 1e-3, 0.005]
    #learning_rates = [1e-1, 1e-2, 0.005]
    #learning_rates = [0.005]
   
    # grid_search(models, optimizers, learning_rates, train_input, train_target, train_class, test_input, test_target, nb_epochs, mini_batch_size)

    # plot_model_comparison(train_input, train_target, train_class, test_input, test_target, optim.SGD, 1e-1, nb_epochs, mini_batch_size)
    plot_netsharing_comparison(train_input, train_target, test_input, test_target, optim.SGD, 1e-1, nb_epochs, mini_batch_size)