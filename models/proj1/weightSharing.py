#!/usr/bin/env python
import torch

from torch import nn
from torch.nn import functional as F


class WeightSharing(nn.Module):
    ''' 
    Input: An image of size 2 x 14 x 14

    Output: Two binary units indicating the image
            with the larger number

    Loss: Binary Cross-Entropy with Logit
    '''
    def __init__(self, nb_hidden=100):
        super(WeightSharing, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.target_type = torch.FloatTensor


    def forward(self, x):
        x1 = x[:, 0, :, :].view(-1, 1, x.shape[2], x.shape[3]) 
        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))
        x1 = x1.view(-1, x1.shape[1]*x1.shape[2]*x1.shape[3])

        x2 = x[:, 1, :, :].view(-1, 1, x.shape[2], x.shape[3])
        x2 = F.relu(F.max_pool2d(self.conv1(x2), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))
        x2 = x2.view(-1, x2.shape[1]*x2.shape[2]*x2.shape[3])

        x = torch.cat((x1, x2), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(-1)


    def predict(self, x):
        return torch.sigmoid(self.forward(x)).round()