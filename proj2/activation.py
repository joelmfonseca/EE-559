import torch
torch.set_grad_enabled(False)

from module import Module

class Tanh(Module):
    '''This class implements the Tanh activation function.'''

    def forward(self, input):
        self.output = input.tanh()
        return self.output

    def backward(self , grad_output):
        return grad_output * (1 - self.output ** 2)

class ReLU(Module):
    '''This class implements the ReLU activation function.'''

    def forward(self, input):
        self.input = input
        return input.clamp(min=0)

    def backward(self, grad_output):
        return grad_output * self.input.sign().clamp(min=0)

class LeakyReLU(Module):
    '''This class implements the LeakyReLU activation function.'''

    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope
    
    def forward(self, input):
        self.input = input
        return input.clamp(min=0) + self.negative_slope*input.clamp(max=0)

    def backward(self, grad_output):
        return grad_output * (self.input.sign().clamp(min=0) - self.input.sign().clamp(max=0).mul(self.negative_slope))

class PReLU(Module):
    '''This class implements the PReLU activation function.'''

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        self.a = torch.empty(num_parameters).fill_(init)
        self.grad_a = torch.empty(num_parameters).zero_()
    
    def forward(self, input):
        self.input = input
        return input.clamp(min=0) + self.a*input.clamp(max=0)

    def backward(self, grad_output):
        if self.num_parameters == 1:
            self.grad_a.add_((grad_output*self.input.clamp(max=0)).sum(0).mean())
        else:
            self.grad_a.add_((grad_output*self.input.clamp(max=0)).sum(0))
        return grad_output * (self.input.sign().clamp(min=0) - self.input.sign().clamp(max=0).mul(self.a))

    def param(self):
        return [(self.a, self.grad_a)]