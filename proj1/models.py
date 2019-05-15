import time

import torch
from torch import nn
from torch.nn import functional as F

class Net1(nn.Module):
    '''This class implements the standard model with Cross Entropy loss.'''
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
    '''This class implements the standard model with Binary Cross Entropy loss.'''
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
    '''This class implements the classification based model with Cross Entropy loss.'''
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
    '''This class implements the weight sharing model with Cross Entropy loss.'''
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
    '''This class implements the weight sharing model with Binary Cross Entropy loss.'''
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
    '''This class implements the weight sharing model with Cross Entropy loss and an additional FC layer.'''
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
    '''This class implements the model with auxiliary loss and the Cross Entropy loss.'''
    def __init__(self, nb_hidden=100):
        super(NetAux1, self).__init__()
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
        x1 = self.fc2(x1)
        
        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x1,x2,x

    def predict(self, x):
        
        _, _ , pred = self.forward(x)
        pred = torch.max(pred,1)[1]
        return pred

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

class NetAux2(nn.Module):
    '''This class implements the model with auxiliary loss and both types of losses.'''
    def __init__(self, nb_hidden=100):
        super(NetAux2, self).__init__()
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
        x1 = self.fc2(x1)
        
        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)

        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x1,x2,x

    def predict(self, x):
        _, _ , pred = self.forward(x)
        return torch.sigmoid(pred).round()
        
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

class NetAux3(nn.Module):
    '''This class implements the model with auxiliary loss, both types of losses and an additional FC layer.'''
    def __init__(self, nb_hidden=100):
        super(NetAux3, self).__init__()
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
        x1 = self.fc2(x1)
        
        x2 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, 256)
        x2 = F.relu(self.fc1(x2))
        x2 = self.fc2(x2)

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