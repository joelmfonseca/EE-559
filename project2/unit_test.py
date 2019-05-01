from loss import CrossEntropyLoss, NLLLoss
from activation import LogSoftmax

import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.Tensor([[-0.3907, -0.3056],
        [ 0.2758, -0.7789],
        [ 1.6004,  0.1921],
        [-0.5768, -0.8816],
        [-1.9635, -0.8913]])

target = torch.Tensor([[0,1],[0,1],[1,0],[0,1],[1,0]])
target_F = torch.Tensor([1, 1, 0, 1, 0]).long()

log_softmax = LogSoftmax()
nll_loss = NLLLoss()

pred = log_softmax.forward(input)
loss = nll_loss.forward(pred, target)
print(pred.exp(), loss)

pred = F.log_softmax(input, dim=1)
loss = F.nll_loss(pred, target_F)
print(pred.exp(), loss)
