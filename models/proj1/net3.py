#!/usr/bin/env python
from torch import nn
from torch.nn import functional as F

import torch

class Net3(nn.Module):
    '''
    Input: An image of size 1 x 14 x 14

    Output: The digit represented on the image

    Loss: Cross-Entropy
    '''
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
