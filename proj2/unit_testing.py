from loss import CrossEntropyLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from module import Linear, Sequential
from activation import Tanh, ReLU, LeakyReLU, PReLU

model = Sequential([
        Linear(2, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 2)]
    )

# input = torch.Tensor([[-0.3907, -0.3056],
#         [ 0.2758, -0.7789],
#         [ 1.6004,  0.1921],
#         [-0.5768, -0.8816],
#         [-1.9635, -0.8913]])

model = nn.Sequential()

input = torch.randn(5, 2, requires_grad=True)

# target = torch.Tensor([[0,1],[0,1],[1,0],[0,1],[1,0]], requires_grad=False)
# target_F = torch.Tensor([1, 1, 0, 1, 0], requires_grad=False).long()

target = torch.empty(5, dtype=torch.long).random_(2)

# cel = CrossEntropyLoss()

# loss = cel.forward(input, target)
# print(loss)

# pred = F.log_softmax(input, dim=1)
# loss = F.nll_loss(pred, target_F)
loss = nn.CrossEntropyLoss()
input = model(Variable(input))
output = loss(input, target)
print(output.grad)
print(output.data)
output.backward()

# print(input)
# print(target_F)
# print(input[target.byte()].view(-1,1))

# - - - - - -  
# from activation import LeakyReLU, ReLU

# lr = LeakyReLU()
# lr = ReLU()
# lr = PReLU()

# res = lr.forward(torch.Tensor([-2, -1, 0, 1, 5]))
# print(res)
# res = lr.backward(0.1)
# print(res)

# nn_relu = nn.ReLU()
# print(nn_relu(torch.Tensor([-2, -1, 0, 1, 5])))

# mse = nn.MSELoss()
# print(torch.max(input,1)[0])
# print(target_F)
# loss = mse(torch.max(input,1)[0], target_F.float())
# print(loss)
# print(loss.backward())