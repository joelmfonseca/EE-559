from module import Module

class Tanh(Module):

    def forward(self, input):
        self.output = input.tanh()
        return self.output

    def backward(self , grad_output):
        return grad_output * (1 - self.output ** 2)

class ReLU(Module):

    def forward(self, input):
        self.input = input
        return input.clamp(min=0)

    def backward(self, grad_output):
        return grad_output * (self.input>0).float()