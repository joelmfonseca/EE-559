import math
import torch
torch.set_grad_enabled(False)

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
        self.grad_weight = torch.empty(self.weight.size()).zero_()
        self.grad_bias = torch.empty(self.bias.size()).zero_()
        self.reset_parameters()

    def reset_parameters(self):

        def xavier_normal_(tensor, gain):
            fan_in, fan_out = self.in_features, self.out_features
            std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            return tensor.normal_(0, std)

        std = 1. / math.sqrt(self.weight.size(1))
        eps = 1e-6
        # self.weight.uniform_(-1, 1)
        # self.weight.normal_(0, eps)
        self.weight = xavier_normal_(self.weight, gain=5.0/3)
        if self.bias is not None:
            # self.bias.uniform_(-1, 1)
            # self.bias.normal_(0, eps)
            self.bias = xavier_normal_(self.bias, gain=5.0/3)

    def forward(self, input):
        self.input = input
        output = input.matmul(self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output

    def backward(self, grad_output):
        self.grad_input = grad_output.matmul(self.weight)

        # print('grad_output: ', grad_output.size())
        # print('input: ', self.input.size())
        # print('weight: ', self.weight.size())
        # print('grad_weight: ', self.grad_weight.size())
        # print('bias: ', self.bias.size())
        # print('grad_bias: ', self.grad_bias.size())

        self.grad_weight.add_(grad_output.t().matmul(self.input))
        if self.bias is not None:
            self.grad_bias.add_(grad_output.sum(0))
        return self.grad_input

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
            input = module.forward(input)
        return input

    def backward(self, grad_output):
        grad = grad_output
        for module in reversed(self.modules):
            grad = module.backward(grad)

    def param(self):
        res = []
        for module in self.modules:
            res.extend(module.param())
        return res

    def load_param(self, param_to_load):
        for i, t in enumerate(self.param()):
            p, _ = t
            p.sub_(p).add_(param_to_load[i][0])

    def print_param(self):
        param = self.param()
        for p, grad in param:
            print('param: ', p)
            print('grad: ', grad)
            print('--')