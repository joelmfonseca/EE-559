from module import Module

class MSELoss(Module):

    def forward(self, input, target):
        self.input = input
        self.target = target
        return (input - target).pow(2).mean()

    def backward(self):
        return (self.input - self.target)