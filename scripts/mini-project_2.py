import torch
torch.set_grad_enabled(False)

import math

class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self , *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features)
        if bias:
            self.bias = torch.empty(out_features)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)

    def forward(self, input):
        self.input = input
        output = input.matmul(self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output

    def backward(self, grad_output):
        self.grad_input, self.grad_weight, self.grad_bias = None, None, None
        self.grad_input = grad_output.matmul(self.weight)
        self.grad_weight = grad_output.t().matmul(self.input)
        if self.bias is not None:
            self.grad_bias = grad_output.sum(0).squeeze(0)
        return self.grad_input, self.grad_weight, self.grad_bias

    def param(self):
        list_param = [(self.weight, self.grad_weight)]
        if self.bias is not None:
            list_param.append((self.bias, self.grad_bias))
        return list_param

class Sequential(Module):

    def __init__(self, modules):
        self.modules = modules

    def forward(self, input):
        for module in self.modules:
            input = module(input)
        return input

    def backward(self, grad_output):
        grad = grad_output
        for module in self.modules:
            grad, _, _ = module.backward(grad)
        return grad

    def param(self):
        return [module.param() for module in self.modules]

class Tanh(Module):

    def forward(self, input):
        self.input = input
        return input.tanh()

    def backward(self , grad_output):
        return grad_output * (1 - self.input.tanh()) ** 2

class ReLU(Module):

    def forward(self, input):
        self.input = input
        return input.clamp(min=0)

    def backward(self, grad_output):
        return grad_output * (self.input>0)

class MSELoss(Module):

    def forward(self, input, target):
        self.input = input
        self.target = target
        return torch.sum((input - target) ** 2)

    def backward(self):
        return 2 * (self.input - self.target)