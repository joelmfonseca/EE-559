from module import Module

class MSELoss(Module):
    '''This class implements the MSE loss.'''

    def forward(self, input, target):
        self.input = input
        self.target = target
        return (input - target).pow(2).mean()

    def backward(self):
        return (self.input - self.target).div_(self.input.size(0))

class CrossEntropyLoss(Module):
    '''This class implements the Cross Entropy loss.'''

    def __nll_loss(self, input, target):
        self.p_k = input[target.byte()].view(-1,1)
        return -input[target.byte()].mean()

    def __log_softmax(self, input):
        denom = input.exp().sum(dim=1).view(-1,1)
        return input.exp().div_(denom).log()

    def forward(self, input, target):
        input = self.__log_softmax(input)
        self.input = input.exp()
        self.target = target
        return self.__nll_loss(input, target)
    
    def backward(self):
        return (self.input - self.target).div_(self.input.size(0))