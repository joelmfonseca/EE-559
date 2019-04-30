class Optimizer(object):

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, param, lr):
        self.param = param
        self.lr = lr

    def step(self):
        for p, grad in self.param:
            if grad is not None:
                p.add_(-self.lr*grad)
                grad.zero_()