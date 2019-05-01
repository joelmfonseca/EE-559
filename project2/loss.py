from module import Module

class MSELoss(Module):

    def forward(self, input, target):
        self.input = input
        self.target = target
        return (input - target).pow(2).mean()

    def backward(self):
        return (self.input - self.target)

class NLLLoss(Module):

    def forward(self, input, target):
        self.input = input
        self.target = target
        return -input[target.byte()].mean()

class CrossEntropyLoss(Module):

    def forward(self, input, target):
        input = log_softmax(input)
        self.input = input
        return NLLLoss.forward(input)
    
    def backward(self):
        return self.input - 1